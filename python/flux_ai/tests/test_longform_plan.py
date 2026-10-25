# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("mido")

from flux_ai import cli  # noqa: E402
from flux_ai.longform import genres, planner, render, symbolic, theory, timing, transitions  # noqa: E402

GENRES = ["deep house", "techno", "lo-fi hip hop", "ambient", "maqam groove"]


def test_genre_tables_load_and_are_marked_as_unsourced():
    assert set(GENRES) <= set(genres.available())
    for name in GENRES:
        g = genres.load(name)
        head = (genres.GENRE_DIR / g["_file"]).read_text().splitlines()[:4]
        assert any("OWNER CURATION" in line for line in head)
    maq = genres.load("bayati")
    assert maq["maqamat"]["bayati"]["cents"][1] == 150  # quarter-tone second degree


@pytest.mark.parametrize("genre", GENRES)
def test_plan_is_deterministic_and_schema_valid(genre):
    a = planner.plan(genre, 1, seed=11)
    b = planner.plan(genre, 1, seed=11)
    c = planner.plan(genre, 1, seed=12)
    assert a.to_json() == b.to_json()
    assert a.to_json() != c.to_json()
    a.validate()
    assert abs(a.duration_seconds() - 3600) < 60  # within a minute of the request
    assert planner.no_repeat_violations(a.data) == []


def test_four_hour_plan_keeps_no_repeat_windows_and_harmonic_moves():
    p = planner.plan("deep house", 4, seed=3)
    assert planner.no_repeat_violations(p.data) == []
    segs = {}
    for s in p.sections:
        segs.setdefault(s["segment"], s)
    codes = [s["camelot"] for s in segs.values()]
    for a, b in zip(codes, codes[1:]):
        assert b in {t for _, t in theory.camelot_neighbours(a)}, (a, b)
    bpms = [s["bpm"] for s in segs.values()]
    assert all(118 <= x <= 125 for x in bpms)
    assert max(abs(x - y) for x, y in zip(bpms, bpms[1:])) <= 2.0


def test_no_repeat_checker_flags_injected_repeat():
    p = planner.plan("techno", 1, seed=5)
    first_of_seg1 = next(s for s in p.sections if s["segment"] == 1)
    first_of_seg0 = p.sections[0]
    first_of_seg1["patterns"]["drums"] = first_of_seg0["patterns"]["drums"]
    assert any("drums" in v for v in planner.no_repeat_violations(p.data))


def test_invalid_plan_is_rejected():
    p = planner.plan("deep house", 0.05, seed=1)
    bad = json.loads(p.to_json())
    bad["sections"][0]["energy"] = 2.0
    with pytest.raises(planner.PlanError):
        planner.validate(bad)


def test_external_minimal_plan_is_accepted(tmp_path: Path):
    plan = {
        "genre": "deep house",
        "sections": [
            {"id": "a", "segment": 0, "type": "groove", "bpm": 120, "key": {"tonic": "A", "mode": "minor"},
             "energy": 0.6, "bars": 8, "seeds": {"section": 1}, "transition_out": {"type": "blend", "bars": 4}},
            {"id": "b", "segment": 1, "type": "peak", "bpm": 124, "key": {"tonic": "E", "mode": "minor"},
             "energy": 0.8, "bars": 8, "seeds": {"section": 2}, "transition_out": {"type": "continue"}},
        ],
    }
    f = tmp_path / "plan.json"
    f.write_text(json.dumps(plan))
    sp = planner.SetPlan.from_file(f)
    assert len(sp.sections[0]["bar_plan"]) == 8
    assert sp.sections[1]["camelot"] == "9A"
    # the tempo ramp in the overlap is computed identically from both sides
    tl = timing.timeline(sp.data)
    assert tl[0].overlap_out_us == tl[1].overlap_in_us > 0
    assert tl[1].start_us == tl[0].start_us + tl[0].dur_us - tl[0].overlap_out_us


def test_midi_is_deterministic_and_carries_tempo_map(tmp_path: Path):
    p = planner.plan("deep house", 0.2, seed=2, constraints={"segment_minutes": 2.0}).data
    i = next(k for k, s in enumerate(p["sections"]) if s["transition_out"]["type"] == "blend")
    p["sections"][i + 1]["bpm"] = p["sections"][i]["bpm"] + 3  # force a tempo ramp
    a, b = tmp_path / "a.mid", tmp_path / "b.mid"
    symbolic.write_midi(p, i, a)
    symbolic.write_midi(p, i, b)
    assert a.read_bytes() == b.read_bytes()
    notes, body_s = render.parse_midi(a)
    expect = sum(timing.section_beat_us(p, i)) / 1e6
    assert abs(body_s - expect) < 1e-6
    assert {n.track for n in notes} >= {"drums"}


def test_maqam_quarter_tones_survive_midi_round_trip(tmp_path: Path):
    p = planner.plan("maqam", 0.2, seed=4, constraints={"start_key": "D bayati"}).data
    # the first groove in bayati (an 8-bar intro may play one motif variation that skips degree 1)
    i = next(k for k, s in enumerate(p["sections"]) if "lead" in s["layers"] and s["key"]["mode"] == "bayati" and s["type"] == "groove")
    out = tmp_path / "m.mid"
    symbolic.write_midi(p, i, out)
    notes, _ = render.parse_midi(out)
    fracs = {round(n.pitch - int(n.pitch), 2) for n in notes if n.track == "lead"}
    assert 0.5 in fracs or -0.5 in fracs  # E half-flat = 150 cents above D


def test_cli_plan_only_emits_protocol_v2(tmp_path: Path, capsys):
    rc = cli.main(["longform", "--genre", "lofi", "--hours", "0.05", "--plan-only", "--out", str(tmp_path), "--json"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert [e["event"] for e in lines] == ["started", "done"]
    assert all(e["protocol_version"] == 2 and e["job_id"] == lines[0]["job_id"] for e in lines)
    assert (tmp_path / "plan.json").exists()
    # global --json flag (v1 style) also works
    rc = cli.main(["--json", "longform", "--genre", "ambient", "--hours", "0.05", "--plan-only", "--out", str(tmp_path)])
    assert rc == 0 and json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "done"


def test_cli_reports_errors_as_events(tmp_path: Path, capsys):
    rc = cli.main(["longform", "--genre", "polka-metal", "--plan-only", "--out", str(tmp_path), "--json"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 1 and lines[-1]["event"] == "error" and "unknown genre" in lines[-1]["message"]


# ---------------------------------------------------------------- arrangement rules (generator v2)

def _notes(plan, i):
    return symbolic.generate_notes(plan.data, i)[0]


def test_new_layers_enter_staggered_after_a_breakdown():
    p = planner.plan("deep house", 0.1666667, seed=1)
    secs = p.sections
    i = next(i for i in range(1, len(secs)) if secs[i - 1]["type"] == "breakdown" and secs[i]["type"] == "build")
    new = [l for l in secs[i]["layers"] if l not in secs[i - 1]["layers"]]
    assert "kick" in new and "bass" in new and "hat_closed" in new
    first = {}
    for n in _notes(p, i):
        key = {v: k for k, v in symbolic.GM_DRUMS.items()}[int(n.pitch)] if n.track == "drums" else n.track
        first[key] = min(first.get(key, 10 ** 9), n.start)
    assert "kick" not in first and "bass" not in first  # held back for the drop
    bars = {l: first[l] // symbolic.BAR for l in new if l in first}
    assert bars["hat_closed"] == 0 and max(bars.values()) >= 1
    assert secs[i]["bar_plan"][-1]["fill"]  # a build ends with a fill
    # a peak right after a build still drops everything at once
    j = next(j for j in range(1, len(secs)) if secs[j - 1]["type"] == "build" and secs[j]["type"] == "peak")
    first = {}
    for n in _notes(p, j):
        if n.track == "drums":
            first[int(n.pitch)] = min(first.get(int(n.pitch), 10 ** 9), n.start)
    assert all(t < symbolic.BAR for t in first.values())


def test_overlap_drums_hand_off_between_sections():
    p = planner.plan("deep house", 0.1666667, seed=1)
    secs = p.sections
    i = next(i for i, s in enumerate(secs) if s["transition_out"]["type"] == "blend")
    tb = secs[i]["transition_out"]["bars"]
    assert secs[i]["transition_out"]["swap_at"] == 0.0
    hat = symbolic.GM_DRUMS["hat_closed"]
    out_bars = {n.start // symbolic.BAR for n in _notes(p, i) if n.track == "drums" and int(n.pitch) == hat}
    in_bars = {n.start // symbolic.BAR for n in _notes(p, i + 1) if n.track == "drums" and int(n.pitch) == hat}
    assert not any(b >= secs[i]["bars"] - tb + tb // 2 for b in out_bars)
    assert not any(b < tb // 2 for b in in_bars)
    assert in_bars and out_bars
    # the low band swaps at the first bar of the overlap
    tl = transitions.Timeline(p.data, [transitions.SectionAudio(k, Path("x"), Path("y"), 1.0, 10) for k in range(len(secs))], 48000)
    assert tl.placed[i + 1].swap_at == tl.placed[i + 1].start


def test_maqam_pad_has_no_third_and_melody_rests_on_tonic_or_ghammaz():
    p = planner.plan("maqam", 0.051, seed=3)
    tonic = theory.pitch_class("D")
    for i, s in enumerate(p.sections):
        for n in _notes(p, i):
            if n.track == "pad":
                assert round((n.pitch - tonic) * 100) % 1200 in (0, 500, 700), n.pitch  # D, G (drone roots) and their fifths
    for m in p.data["motifs"]:
        assert m["notes"][-1][2] in (0, 3)
    degs = [n[2] for m in p.data["motifs"] for n in m["notes"]]
    assert degs.count(2) >= len(degs) * 0.1  # F (degree 2) is part of the tune


def test_set_ends_with_a_fade():
    p = planner.plan("techno", 0.05, seed=2)
    n = len(p.sections)
    tl = transitions.Timeline(p.data, [transitions.SectionAudio(k, Path("x"), Path("y"), 1.0, 10 ** 8) for k in range(n)], 48000)
    last = tl.placed[-1]
    rel = np.array([tl.fade_from - last.start - 1, (tl.fade_from + tl.end_of_body) // 2 - last.start, tl.end_of_body - last.start - 1])
    gl, gh = tl.gains(last, rel)
    assert gh[0] == 1.0 and 0.5 < gh[1] < 0.9 and gh[2] < 0.01
    assert np.allclose(gl, gh)


def test_flux_host_layer_eq_comes_from_the_genre_table():
    from flux_ai.longform.fluxhost import FluxHostRenderer

    r = FluxHostRenderer(engine="/nonexistent")
    surge = {"name": "Surge XT"}
    eq = r.layer_eq(genres.load("deep house"), {"bass": surge, "chords": surge, "lead": surge, "pad": surge, "drums": None})
    assert eq["chords"] == {"highpass": 200.0, "lowpass": 1800.0} and "bass" not in eq
    assert r.layer_eq(genres.load("techno"), {"chords": surge}) == {}


def test_build_section_carries_a_riser_in_the_builtin_synth(tmp_path: Path):
    p = planner.plan("deep house", 0.1666667, seed=1).data
    i = next(i for i, s in enumerate(p["sections"]) if s["type"] == "build")
    r = render.BuiltinSynthRenderer(sample_rate=12000, tail_seconds=0.5)
    notes, body_s = [], p["sections"][i]["bars"] * 4 * 60 / p["sections"][i]["bpm"]
    quiet = r.synthesize(notes, body_s, {**p["sections"][i], "type": "groove"}, tracks={"drums"})
    riser = r.synthesize(notes, body_s, p["sections"][i], tracks={"drums"})
    assert not np.any(quiet.high)
    b = riser.body_frames
    e_first, e_last = np.mean(riser.high[: b // 4] ** 2), np.mean(riser.high[3 * b // 4: b] ** 2)
    assert e_last > 20 * e_first > 0


def test_drum_voices_route_to_their_own_engine_tracks():
    from flux_ai.longform import fluxhost

    notes = [render.ParsedNote("drums", 9, 0.0, 0.1, 36, 0.9, 0.5), render.ParsedNote("drums", 9, 0.5, 0.6, 39, 0.8, 0.5),
             render.ParsedNote("drums", 9, 0.25, 0.3, 42, 0.5, 0.5), render.ParsedNote("bass", 0, 0.0, 0.5, 45.5, 0.7, 0.5)]
    routing = {"bass": "bass", "drum:kick": "drum:kick", "drum:clap": "drum:clap"}
    clips, stats = fluxhost.notes_to_clips(notes, 120.0, 4.0, routing, {"kick": 48, "clap": 60})
    assert [n["pitch"] for n in clips["drum:kick"]] == [48] and clips["drum:kick"][0]["velocity"] == 114
    assert [n["pitch"] for n in clips["drum:clap"]] == [60] and clips["drum:clap"][0]["cents"] == 0
    assert "drum:hat_closed" not in clips and stats["notes"] == 3  # the hat stays on the built-in synth
    assert abs(clips["bass"][0]["cents"]) == 50  # 45.5 rounds to 46 with -50 cents
    r = fluxhost.FluxHostRenderer(engine="/nonexistent")
    voices = r.drum_presets(genres.load("deep house"), {"name": "Surge XT"})
    assert voices["kick"]["note"] == 48 and voices["clap"]["preset"].endswith("Clap - Noise Layer")
    assert r.drum_presets(genres.load("techno"), {"name": "Surge XT"}) == {}
    r.plugins["drums"] = "Some Kit"
    assert r.drum_presets(genres.load("deep house"), {"name": "Surge XT"}) == {}

