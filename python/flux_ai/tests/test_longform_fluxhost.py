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
from flux_ai.longform.fluxhost import FluxHostRenderer, notes_to_clips, parse_plugin_specs  # noqa: E402
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
    assert st["microtones_rounded"] == 1 and st["clamped"] == 1 and clips["pad"] == []
    clips, _ = notes_to_clips(notes, 120.0, 8, {"lead": "lead"})  # unrouted layers are skipped
    assert list(clips) == ["lead"]


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
