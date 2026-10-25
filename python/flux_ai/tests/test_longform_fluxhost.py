# SPDX-License-Identifier: GPL-3.0-or-later
"""FluxHostRenderer: MIDI -> write_notes conversion, plugin selection, decision log, and an end-to-end
run through flux-engine with the bundled CLAP "Test Synth" (skipped when either is not found)."""
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

for mod in ("scipy", "soundfile", "pyloudnorm", "mido", "jsonschema", "mutagen", "yaml"):
    pytest.importorskip(mod)

import pyloudnorm  # noqa: E402
import soundfile as sf  # noqa: E402

from flux_ai.longform import analysis, master, outputs, pipeline, planner, symbolic, timing  # noqa: E402
from flux_ai.longform.fluxhost import (  # noqa: E402
    FluxHostRenderer, base_layer, notes_to_clips, parse_plugin_specs, parse_preset_specs, split_microtones)
from flux_ai.longform.render import ParsedNote, get_renderer, parse_midi  # noqa: E402

HERE = Path(__file__).resolve()


def two_section_plan(seed: int = 21) -> dict:
    """About 3 minutes: two full-arrangement sections, an 8-bar blend with bass swap and a +3 BPM ramp."""
    p = planner.plan("deep house", 0.05, seed=seed, constraints={"segment_minutes": 1.6, "transition_bars": 8}).data
    a = copy.deepcopy(next(s for s in p["sections"] if s["segment"] == 0 and s["type"] == "peak"))
    b = copy.deepcopy(next(s for s in p["sections"] if s["segment"] == 1 and s["type"] == "peak"))
    a.update(id="s000", bars=44, transition_out={"type": "blend", "bars": 8, "eq": True, "bass_swap": True})
    b.update(id="s001", bars=52, bpm=a["bpm"] + 3, transition_out={"type": "continue", "bars": 0})
    for s in (a, b):
        s.pop("bar_plan", None)
    p["sections"] = [a, b]
    return planner.normalise(p)


# ------------------------------------------------------------------ conversion

def test_plugin_specs():
    assert parse_plugin_specs(["bass=Surge XT", "default = Test Synth"]) == {"bass": "Surge XT", "default": "Test Synth"}
    assert parse_plugin_specs(["lead=CLAP-x=y"]) == {"lead": "CLAP-x=y"}
    for bad in (["bass"], ["vocals=X"], ["bass="]):
        with pytest.raises(ValueError):
            parse_plugin_specs(bad)


def test_midi_to_write_notes_follows_the_tempo_ramp(tmp_path):
    plan = two_section_plan()
    midi = tmp_path / "s000.mid"
    symbolic.write_midi(plan, 0, midi)
    notes, body_s = parse_midi(midi)
    bpm = float(plan["sections"][0]["bpm"])
    clip_beats = 4 * 46
    clips, st = notes_to_clips(notes, bpm, clip_beats, {"bass": "bass", "chords": "chords", "pad": "pad", "lead": "lead"})
    pitched = [n for n in notes if n.track in ("bass", "chords", "pad", "lead")]
    assert st["notes"] == len(pitched) == sum(len(v) for v in clips.values()) and st["clamped"] == 0
    assert set(clips) == {"bass", "chords", "pad", "lead"} and all(clips.values())
    # positions are the MIDI's absolute seconds (tempo ramp included) at the constant project tempo
    by_track = {k: iter(v) for k, v in clips.items()}
    for n in pitched:
        w = next(by_track[n.track])
        assert w["start_beat"] == round(n.start * bpm / 60 * 960) / 960
        assert abs(w["start_beat"] + w["length_beats"] - n.end * bpm / 60) <= 1 / 960
        assert 1 <= w["velocity"] <= 127 and w["velocity"] == max(1, round(n.vel * 127))
        assert w["pitch"] == round(n.pitch) and w["length_beats"] >= 1 / 960
        assert w["start_beat"] + w["length_beats"] <= clip_beats
    # the outgoing ramp really is in the MIDI: the last bar is shorter than at the section tempo
    pl = timing.timeline(plan)[0]
    last_bar_s = (pl.bar_starts_us[-1] - pl.bar_starts_us[-2]) / 1e6
    assert last_bar_s < 4 * 60 / bpm - 1e-3
    assert abs(body_s - pl.dur_us / 1e6) < 1e-3


def test_drum_split_microtones_and_clip_end():
    notes = [ParsedNote("drums", 9, 0.0, 0.1, 36, 1.0, 0.5), ParsedNote("drums", 9, 0.5, 0.6, 42, 0.5, 0.5),
             ParsedNote("lead", 0, 1.0, 9.0, 62.5, 0.004, 0.5), ParsedNote("pad", 1, 30.0, 31.0, 60, 0.5, 0.5)]
    clips, st = notes_to_clips(notes, 120.0, 8, {"drums": "drums", "kick": "kick", "lead": "lead", "pad": "pad"})
    assert [n["pitch"] for n in clips["kick"]] == [36] and [n["pitch"] for n in clips["drums"]] == [42]
    lead = clips["lead"][0]
    assert lead["pitch"] in (62, 63) and lead["velocity"] == 1
    assert lead["start_beat"] == 2.0 and lead["start_beat"] + lead["length_beats"] == 8  # cut at the clip end
    assert lead["cents"] in (50, -50)  # the quarter tone is kept as an offset, not rounded away
    assert st["microtonal_notes"] == 1 and st["clamped"] == 1 and clips["pad"] == []
    clips, _ = notes_to_clips(notes, 120.0, 8, {"lead": "lead"})  # unrouted layers are skipped
    assert list(clips) == ["lead"]



def test_split_microtones_routes_each_offset_to_its_own_bent_track():
    n = lambda p, c: {"pitch": p, "start_beat": 0, "length_beats": 1, "velocity": 100, "cents": c}
    clips = {"lead": [n(64, -50), n(62, 0), n(64, -50), n(69, 50)], "pad": [n(60, 0)], "bass": [n(40, 250)]}
    out, events, st = split_microtones(clips, {"lead": 2.0, "bass": 2.0})
    assert sorted(out) == ["bass", "lead", "lead~+50c", "lead~-50c", "pad"]
    assert [x["pitch"] for x in out["lead~-50c"]] == [64, 64] and len(out["lead"]) == 1
    assert events["lead~-50c"] == [{"type": "pitch_bend", "beat": 0, "value": -0.25}]
    assert events["lead~+50c"][0]["value"] == 0.25 and "lead" not in events and "pad" not in events
    assert st == {"microtone_tracks": 2, "microtonal_notes_carried": 3, "microtones_out_of_range": 1}
    assert all("cents" not in x for xs in out.values() for x in xs)
    assert base_layer("lead~-50c") == "lead" and base_layer("pad") == "pad"
    _, ev, _ = split_microtones({"lead": [n(64, -50)]}, {"lead": 1.0})  # a ±1 st instrument bends twice as far
    assert ev["lead~-50c"][0]["value"] == -0.5


def test_plugin_precedence(monkeypatch):
    monkeypatch.delenv("FLUX_LONGFORM_PLUGIN", raising=False)
    g = {"flux_host": {"plugins": {"pad": "Genre Pad", "default": "Genre Default"}}}
    r = FluxHostRenderer(plugins={"bass": "Cli Bass"})
    assert r.layer_specs(g) == {"drums": None, "bass": "Cli Bass", "chords": "Genre Default", "pad": "Genre Pad",
                                "lead": "Genre Default"}
    r = FluxHostRenderer(plugins={"default": "Cli Default", "drums": "Kit"})
    assert r.layer_specs(g)["chords"] == "Cli Default" and r.layer_specs(g)["drums"] == "Kit"
    assert FluxHostRenderer().layer_specs({})["lead"] == "Surge XT"
    monkeypatch.setenv("FLUX_LONGFORM_PLUGIN", "Env Synth")
    assert FluxHostRenderer().layer_specs({})["lead"] == "Env Synth"
    gains = FluxHostRenderer().layer_gains({"timbre": {"bass": {"level_db": -11}, "pad": {"level_db": -25}}})
    assert gains["bass"] == 0 and gains["pad"] == -14 and gains["drums"] == 0



def test_layer_presets_and_per_plugin_gains():
    assert parse_preset_specs(["bass=Deep end", "pad = Pad 1"]) == {"bass": "Deep end", "pad": "Pad 1"}
    with pytest.raises(ValueError):
        parse_preset_specs(["default=Pad 1"])  # presets are per layer
    genre = {"timbre": {"bass": {"level_db": -10}, "pad": {"level_db": -20}},
             "flux_host": {"presets": {"Surge XT": {"bass": "Deep end", "pad": "Winter warmer"}},
                           "gain_db": {"Surge XT": {"bass": 0.0, "pad": -7.5}}}}
    surge, other = {"name": "Surge XT", "id": "s"}, {"name": "Test Synth", "id": "t"}
    resolved = {"bass": surge, "pad": surge, "lead": other, "chords": surge, "drums": None}
    r = FluxHostRenderer(presets={"pad": "Pad 1"})
    pre = r.layer_presets(genre, resolved)
    assert pre["bass"] == "Deep end"            # genre table for the layer's plugin
    assert pre["pad"] == "Pad 1"                # --flux-preset wins
    assert pre["lead"] is None                  # no table for Test Synth: default patch
    assert pre["chords"] is None and pre["drums"] is None
    gains = r.layer_gains(genre, resolved)
    assert gains["pad"] == -7.5                 # measured Surge XT offset
    assert gains["lead"] == 0.0                 # other plugin: timbre balance (lead has no level: same as bass)
    flat = r.layer_gains({"flux_host": {"gain_db": {"pad": -3}}}, {"pad": other})
    assert flat["pad"] == -3                    # a flat map applies to any plugin
    genre["flux_host"]["transpose"] = {"Surge XT": {"bass": 24}}
    shift = r.layer_transpose(genre, resolved)
    assert shift["bass"] == 24 and shift["pad"] == 0 and shift["lead"] == 0 and shift["drums"] == 0


def test_genre_tables_name_surge_presets_for_every_pitched_layer():
    from flux_ai.longform import genres
    for name in ("deep house", "techno", "lo-fi hip hop", "ambient", "maqam"):
        g = genres.load(name)
        presets = g["flux_host"]["presets"]["Surge XT"]
        gains = g["flux_host"]["gain_db"]["Surge XT"]
        shift = g["flux_host"]["transpose"]["Surge XT"]
        assert set(presets) == set(gains) == set(shift) == {"bass", "chords", "pad", "lead"}, name
        assert all(v % 12 == 0 for v in shift.values()), name  # octave corrections only


def test_click_attribution(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    x = 0.3 * np.sin(2 * np.pi * 110 * t)
    x[sr // 2 + 100:] = 0  # a hard note-off (at a non-zero phase) inside the section render
    src = tmp_path / "a.flac"
    sf.write(str(src), np.c_[x, x], sr, subtype="PCM_24")
    sources = [(src, 1000, sr)]
    k = int(analysis.find_discontinuities(np.c_[x, x], sr)[0])
    assert analysis.click_in_sources(1000 + k, sources, sr)
    assert not analysis.click_in_sources(1000 + sr // 4, sources, sr)  # not in the source: the join made it
    assert not analysis.click_in_sources(10, sources, sr)


def test_decision_log_collects_engine_turns():
    plan = {"human_decisions": [{"what": "genre", "value": "deep house"}]}
    metas = [{"id": "s000", "stats": {"engine": {"project": "project", "history": {"count": 2},
                                                 "turns": [{"label": "Agent: longform s000: instruments", "recorded": True},
                                                           {"label": "Agent: longform s000: notes", "recorded": True}]}}},
             {"id": "s001", "stats": {}}]
    log = outputs.decision_log(plan, metas, [Path("sections/s000-x"), Path("sections/s001-y")])
    assert [e["actor"] for e in log["entries"]] == ["human", "agent", "agent"]
    assert log["entries"][2]["turn"] == "Agent: longform s000: notes"
    assert log["engine_history"] == {"s000": {"get_history": {"count": 2}, "project": "sections/s000-x/project"}}
    assert log["human_review"] is None and "human-review" in log["note"]


# ----------------------------------------------------------------- integration

def _find_up(rel: str) -> Path | None:
    for parent in HERE.parents:
        if (parent / rel).exists():
            return parent / rel
    return None


def _engine_env() -> tuple[str, str] | None:
    eng = os.environ.get("FLUX_ENGINE") or shutil.which("flux-engine") or \
        str(_find_up("flux-host/conanbuild/Release/products/bin/flux-engine") or "")
    clap = os.environ.get("FLUX_CLAP_PATH") or str(_find_up("flux-host/conanbuild/Release/tests/test_plugins/CLAP/Release") or "")
    if not eng or not os.access(eng, os.X_OK) or not clap:
        return None
    return eng, clap


class CountingFluxHost(FluxHostRenderer):
    calls = 0

    def render(self, section_midi, section_plan, out_wav):
        type(self).calls += 1
        return super().render(section_midi, section_plan, out_wav)


@pytest.fixture(scope="module")
def flux_run(tmp_path_factory):
    env = _engine_env()
    if env is None:
        pytest.skip("flux-engine or the CLAP Test Synth not found: set FLUX_ENGINE (flux-host target flux-engine) "
                    "and FLUX_CLAP_PATH (flux-host tests/test_plugins/CLAP/Release)")
    eng, clap = env
    mp = pytest.MonkeyPatch()
    mp.setenv("FLUX_CLAP_PATH", clap)
    r = get_renderer("flux-host", engine=eng, plugins={"default": "Test Synth"})
    ok, why = r.available()
    if not ok:
        r.close()
        mp.undo()
        pytest.skip(f"flux-host renderer unavailable with Test Synth: {why}")
    out = tmp_path_factory.mktemp("lfh") / "run"
    try:
        report = pipeline.run(two_section_plan(), out, renderer=r)
    finally:
        r.close()
    yield out, report, eng
    mp.undo()


def test_flux_host_master_in_spec(flux_run):
    out, rep, _ = flux_run
    x, sr = sf.read(rep["outputs"]["master"], dtype="float64")
    assert sr == 48000 and 150 < len(x) / sr < 240
    assert abs(pyloudnorm.Meter(sr).integrated_loudness(x) + 14.0) <= 0.5
    assert master.true_peak_dbtp(x) <= -1.0
    assert rep["loudness_in_spec"] and rep["true_peak_in_spec"]
    # the blend's joins add no click (Test Synth's own hard note-offs are counted separately)
    assert rep["joins_checked"] == 1 and rep["join_clicks"] == 0


def test_flux_host_stems_are_not_silent(flux_run):
    out, rep, _ = flux_run
    assert rep["renderer"]["name"] == "flux-host" and rep["sections_rendered"] == 2
    for d in sorted((out / "sections").iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        eng = meta["stats"]["engine"]
        assert set(eng["stem_rms_dbfs"]) == {"bass", "chords", "pad", "lead"}
        assert all(v > -60 for v in eng["stem_rms_dbfs"].values()), eng["stem_rms_dbfs"]
        for f in ("mix.flac", "mix.low.flac"):
            y = sf.read(str(d / f))[0]
            assert np.sqrt(np.mean(y ** 2)) > 1e-3, (d.name, f)
        assert (d / "project" / "project.zpj").is_file()  # the saved engine project, for human review
        assert meta["stats"]["plugins"]["drums"] == "builtin-synth"
        assert meta["stats"]["plugins"]["bass"]["name"] == "Test Synth"


def test_flux_host_provenance(flux_run):
    out, rep, _ = flux_run
    prov = json.loads((out / "provenance.json").read_text())
    rend = prov["components"][2]
    assert rend["name"] == "flux-host" and rend["engine"]["api_version"] == "0.1"
    assert rend["plugins"]["default"]["name"] == "Test Synth"
    assert set(prov["section_rendering"]) == {"s000", "s001"}
    turns = [e["turn"] for e in prov["decision_log"]["entries"] if e["actor"] == "agent"]
    assert turns == [f"Agent: longform {s}: {w}" for s in ("s000", "s001") for w in ("instruments", "notes")]
    assert prov["decision_log"]["engine_history"]["s001"]["get_history"]["count"] == 2
    assert "Test Synth" in prov["ddex_style_ai_disclosure"]["instrumentation"]


def test_flux_host_resume_renders_nothing(flux_run):
    out, rep, eng = flux_run
    CountingFluxHost.calls = 0
    r = CountingFluxHost(engine=eng, plugins={"default": "Test Synth"})
    try:
        rep2 = pipeline.run(json.loads((out / "plan.json").read_text()), out, renderer=r)
    finally:
        r.close()
    assert CountingFluxHost.calls == 0 and rep2["sections_cached"] == 2 and rep2["sections_rendered"] == 0
    a = sf.read(rep["outputs"]["master"])[0]
    assert np.allclose(a, sf.read(rep2["outputs"]["master"])[0], atol=1e-6)


def test_flux_host_restarts_a_crashed_engine(flux_run, tmp_path):
    out, rep, eng = flux_run
    plan = json.loads((out / "plan.json").read_text())
    r = FluxHostRenderer(engine=eng, plugins={"default": "Test Synth"})
    try:
        r.identity()
        first = r.client._proc
        first.kill()  # the engine dies between sections
        first.wait()
        (tmp_path / "a").mkdir()
        res = r.render(out / "midi" / "s000.mid", plan["sections"][0], tmp_path / "a" / "mix.flac")
        assert r.client._proc is not first and r.client.running
        assert res.frames > res.body_frames and (tmp_path / "a" / "mix.low.flac").is_file()
    finally:
        r.close()


# ---------------------------------------------------------------- Surge XT tuning and FX helpers

def test_scala_slots_scl_kbm_and_pitch_mapping(tmp_path):
    from flux_ai.longform import surge_tuning as st

    bayati = [0, 150, 300, 500, 700, 800, 1000]
    slots = st.scale_slots(bayati)
    assert slots[2] == 150.0 and slots[1] == 100.0 and slots[3] == 300.0 and len(slots) == 12
    # written pitches (D4 tonic = 62) land on the chromatic slot that is retuned to them
    assert [st.pitch_to_slot_note(p, 62, slots) for p in (62, 63.5, 65, 67, 74, 75.5, 61, 51.5)] == [62, 64, 65, 67, 74, 76, 61, 52]
    scl = st.scl_text(slots, "flux D bayati")
    lines = [l for l in scl.splitlines() if l and not l.startswith("!")]
    assert lines[0] == "flux D bayati" and lines[1].strip() == "12" and lines[3].strip() == "150.00000" and lines[-1].strip() == "1200.00000"
    kbm = st.kbm_text(62, 293.6648)
    vals = [l for l in kbm.splitlines() if l and not l.startswith("!")]
    assert vals[:6] == ["12", "0", "127", "62", "62", "293.664800"] and vals[6] == "12" and vals[7:] == [str(i) for i in range(12)]
    with pytest.raises(ValueError):
        st.scale_slots([0, 150, 160])  # two degrees on one chromatic slot
    assert abs(st.fx_type_value("Reverb 1") - 2 / 29) < 1e-9 and st.fx_type_value("off") == 0.0
    with pytest.raises(KeyError):
        st.fx_type_value("Plate")
    # user default file: enable writes both keys and keeps other entries
    p = tmp_path / "SurgeXTUserDefaults.xml"
    p.write_text('<?xml version = "1.0" encoding = "UTF-8" ?>\n<defaults version="1">\n  <default key="other" value="x" type="1"/>\n</defaults>\n')
    assert not st.patch_tuning_enabled(p)
    st.enable_patch_tuning(p)
    assert st.patch_tuning_enabled(p) and 'key="other"' in p.read_text()


def test_tuned_patch_rewrites_a_surge_fxp(tmp_path):
    import struct
    from xml.etree import ElementTree as ET

    from flux_ai.longform import surge_tuning as st

    src = st.find_preset_file("Surge XT Factory Presets/Organ 1")
    if src is None:
        pytest.skip("Surge XT factory patches not installed")
    assert st.find_preset_file("Surge XT Factory Presets/Pianobass").name == "Piano Bass.fxp"  # catalogue name vs file name
    out = st.tuned_patch(src, tmp_path / "t.fxp", st.scl_text(st.scale_slots([0, 150, 300, 500, 700, 800, 1000]), "t"),
                         st.kbm_text(62, 293.66), "flux D bayati")
    raw = out.read_bytes()
    assert raw[:4] == b"CcnK" and raw[60:64] == b"sub3"
    xml_size = struct.unpack("<I", raw[64:68])[0]
    chunk_size = struct.unpack(">i", raw[56:60])[0]
    assert 60 + chunk_size == len(raw) and 92 + xml_size <= len(raw)
    root = ET.fromstring(st.patch_xml(out))
    pt = root.find("patchTuning")
    assert pt is not None and pt.get("mname") == "flux D bayati"
    import base64
    assert base64.b64decode(pt.get("v")).decode().splitlines()[2] == "t"
    # a second embedding replaces the first
    out2 = st.tuned_patch(out, tmp_path / "t2.fxp", "! x\nx\n 12\n!\n 100.0\n 200.0\n 300.0\n 400.0\n 500.0\n 600.0\n 700.0\n 800.0\n 900.0\n 1000.0\n 1100.0\n 1200.0\n")
    assert st.patch_xml(out2).count(b"<patchTuning") == 1


def test_renderer_tuning_mode_fx_table_and_pitch_map(monkeypatch):
    from flux_ai.longform import fluxhost, genres
    from flux_ai.longform import surge_tuning as st

    r = fluxhost.FluxHostRenderer(engine="/nonexistent")
    assert r.tuning_mode(genres.load("maqam")) == "scala" and r.tuning_mode(genres.load("deep house")) == "bend"
    surge = {"name": "Surge XT"}
    fx = r.layer_fx(genres.load("deep house"), {"pad": surge, "chords": surge, "lead": surge, "bass": surge, "drums": None})
    assert fx["pad"] == [{"slot": "G1", "type": "Reverb 1", "params": {}}] and "lead" not in fx
    slots = st.scale_slots([0, 150, 300, 500, 700, 800, 1000])
    notes = [ParsedNote("lead", 1, 0.0, 0.5, 63.5, 0.8, 0.5), ParsedNote("lead", 1, 0.5, 1.0, 62.0, 0.8, 0.5),
             ParsedNote("drums", 9, 0.0, 0.1, 36, 0.9, 0.5)]
    clips, _ = fluxhost.notes_to_clips(notes, 120.0, 4.0, {"lead": "lead", "drums": "drums"}, None,
                                       lambda p: st.pitch_to_slot_note(p, 62, slots))
    assert [n["pitch"] for n in clips["lead"]] == [64, 62] and all(n["cents"] == 0 for n in clips["lead"])
    assert clips["drums"][0]["pitch"] == 36


def _engine_and_surge():
    """(engine path, Surge XT plugin id) or None when either is missing."""
    eng = os.environ.get("FLUX_ENGINE") or shutil.which("flux-engine")
    if not eng or not Path(eng).exists():
        return None
    from flux_ai.longform.fluxhost import FluxHostRenderer

    r = FluxHostRenderer(engine=eng)
    try:
        surge = next((p for p in r.catalog() if p.get("name") == "Surge XT"), None)
    except Exception:
        surge = None
    finally:
        r.close()
    return (eng, surge["id"]) if surge else None


def test_embedded_scale_retunes_surge_note(tmp_path):
    """A patch with an embedded bayati scale plays MIDI 64 (E4) at E half-flat, 150 cents above D4."""
    es = _engine_and_surge()
    if es is None:
        pytest.skip("flux-engine or Surge XT not found")
    from flux_ai.longform import surge_tuning as st
    from flux_ai import engine_client

    if not st.patch_tuning_enabled():
        pytest.skip("Surge XT user default overrideTuningOnPatchLoad is off")
    src = st.find_preset_file("Surge XT Factory Presets/Init Saw") or st.find_preset_file("Surge XT Factory Presets/Init Sine")
    if src is None:
        pytest.skip("no Surge init patch found")
    eng, surge = es
    slots = st.scale_slots([0, 150, 300, 500, 700, 800, 1000])
    tuned = st.tuned_patch(src, tmp_path / "bayati.fxp", st.scl_text(slots, "flux D bayati"), st.kbm_text(62, 293.6648), "flux D bayati")
    c = engine_client.EngineClient(eng, ["--sr", "48000"])
    try:
        c.call("create_project", {"dir": str(tmp_path / "proj"), "bpm": 120}, 60)
        t = c.call("add_track", {"type": "instrument", "name": "s", "plugin_id": surge}, 60)
        got = c.call("load_preset", {"plugin": t["plugin"]["id"], "preset": str(tuned)}, 60)["preset"]
        assert got.get("source") == "file"
        hz = {}
        for note in (62, 64, 65):
            c.call("write_notes", {"track": t["track"], "start_bar": 1, "length_bars": 1,
                                   "notes": [{"pitch": note, "start_beat": 0, "length_beats": 2, "velocity": 100}]}, 60)
            c.call("set_range", {"start_bar": 1, "end_bar": 2}, 60)
            out = tmp_path / f"n{note}.wav"
            c.call("render", {"out": str(out), "start_bar": 1, "end_bar": 2, "bits": "32f"}, 300)
            x, sr = sf.read(str(out), dtype="float32", always_2d=True)
            x = x[int(0.2 * sr):int(0.9 * sr)].mean(1)
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=1 << 18))
            f = np.fft.rfftfreq(1 << 18, 1 / sr)
            band = (f > 200) & (f < 450)
            hz[note] = float(f[band][np.argmax(spec[band])])
    finally:
        c.close()
    cents = {n: 1200 * np.log2(hz[n] / 293.6648) for n in hz}
    assert abs(cents[62]) < 8, cents
    assert abs(cents[64] - 150) < 8, cents  # the retuned E: a quarter tone below E natural (200)
    assert abs(cents[65] - 300) < 8, cents


def test_brightness_curve_and_automation_table():
    from flux_ai.longform import fluxhost, genres

    p = planner.plan("deep house", 0.1666667, seed=1).data
    sec = next(s for s in p["sections"] if s["type"] == "peak")
    curve = symbolic.brightness_curve(sec)
    assert len(curve) == sec["bars"] and all(0.0 <= v <= 1.0 for v in curve)
    assert max(curve) - min(curve) > 0.05  # the drift and the energy slope move it
    notes, extra = symbolic.generate_notes(p, p["sections"].index(sec))
    assert [v for _, _, v in extra["cc"]] == [int(round(v * 127)) for v in curve]
    r = fluxhost.FluxHostRenderer(engine="/nonexistent")
    surge = {"name": "Surge XT"}
    auto = r.layer_automation(genres.load("deep house"), {"pad": surge, "chords": surge, "lead": surge, "bass": surge, "drums": None})
    assert auto["pad"] == [{"param": "A Filter 1 Cutoff", "depth": 0.12}] and "lead" not in auto


def test_automation_reaches_the_engine_when_a_table_names_the_plugin(tmp_path, monkeypatch):
    """With the Test Synth standing in for Surge XT, the renderer writes one automation clip per swept layer."""
    es = os.environ.get("FLUX_ENGINE") or shutil.which("flux-engine")
    if not es or not Path(es).exists():
        pytest.skip("flux-engine not found")
    from flux_ai.longform import fluxhost, genres

    r = fluxhost.FluxHostRenderer(engine=es, plugins={"default": "Test Synth"})
    ok, why = r.available()
    if not ok:
        r.close()
        pytest.skip(why)
    try:
        g = copy.deepcopy(genres.load("deep house"))
        g["flux_host"]["automation"] = {"Test Synth": {"pad": [{"param": "Gain", "depth": 0.3}]}}  # Gain sits mid-range; Level starts at its maximum
        g["flux_host"]["presets"] = {}
        g["flux_host"]["drum_presets"] = {}
        g["flux_host"]["fx"] = {}
        g["flux_host"]["eq"] = {}
        monkeypatch.setattr(genres, "load", lambda name, _g=g: _g)
        p = planner.plan("deep house", 0.02, seed=1).data
        i = next(k for k, s in enumerate(p["sections"]) if "pad" in s["layers"])
        midi = tmp_path / "s.mid"
        symbolic.write_midi(p, i, midi)
        res = r.render(midi, p["sections"][i], tmp_path / "out" / "mix.flac")
        auto = res.stats["engine"]["automation"]
        assert "pad" in auto and auto["pad"][0]["param"] == "Gain" and auto["pad"][0]["max"] > auto["pad"][0]["min"]
        assert res.stats["automation"] == {"pad": ["Gain"]}
    finally:
        r.close()

