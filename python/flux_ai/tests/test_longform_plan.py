# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("mido")

from flux_ai import cli  # noqa: E402
from flux_ai.longform import genres, planner, render, symbolic, theory, timing  # noqa: E402

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
    i = next(k for k, s in enumerate(p["sections"]) if "lead" in s["layers"] and s["key"]["mode"] == "bayati")
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
