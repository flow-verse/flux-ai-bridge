# SPDX-License-Identifier: GPL-3.0-or-later
"""A full MCP session against the real flux-engine: edits, turns/undo, render, AI worker and error paths.

Skipped (with the reason) when flux-engine or the CLAP Test Synth is not available.
"""
import json
from pathlib import Path

import pytest

from mcp import Client

from flux_mcp.server import Config, Session, build_server

pytestmark = pytest.mark.anyio


class S:
    """Small helper around an MCP client."""

    def __init__(self, client):
        self.c = client

    async def ok(self, name, /, **args):
        r = await self.c.call_tool(name, args)
        assert not r.is_error, f"{name} failed: {r.content[0].text}"
        return r.structured_content

    async def err(self, name, /, **args) -> str:
        r = await self.c.call_tool(name, args)
        assert r.is_error, f"{name} unexpectedly succeeded"
        return r.content[0].text


@pytest.fixture
async def flux(engine_env, tmp_path):
    server = build_server(Session(Config(workdir=tmp_path)))
    async with Client(server) as c:
        yield S(c), server.session, tmp_path


def _clip_ids(proj):
    return {c["id"] for t in proj["tracks"] for c in t.get("clips", [])}


async def test_full_session(flux):
    s, session, tmp = flux
    assert (await s.ok("get_project"))["open"] is False
    created = await s.ok("new_project", dir="song", bpm=124)
    assert created["project"]["dir"] == str(tmp / "song")
    plugins = (await s.ok("find_plugins", instruments_only=True))["plugins"]
    assert any(p["name"] == "Test Synth" for p in plugins)

    r = await s.ok("add_track", name="Bass", plugin="Test Synth")
    track = r["track"]
    assert r["undo_step"] == "Agent: add instrument track Bass"
    assert any(t["id"] == track for t in r["project"]["tracks"])  # fresh summary after every edit
    plugin = r["results"][0]["plugin"]["id"]

    r = await s.ok("write_midi_clip", track_id=track, start_bar=1, length_bars=2,
                   notes=[{"pitch": "C2", "start_beat": 0, "length_beats": 1},
                          {"pitch": 43, "start_beat": 4, "length_beats": 2, "velocity": 90}])
    clip = r["clip"]
    proj = await s.ok("get_project", include_notes=True)
    notes = next(c for t in proj["tracks"] for c in t["clips"] if c["id"] == clip)["notes"]
    assert [n["pitch"] for n in notes] == [36, 43]
    assert proj["unsaved_changes"] is True

    r = await s.ok("copy_clip", clip_id=clip, to_bar=3, transpose_semitones=12)
    proj = await s.ok("get_project", include_notes=True)
    copy = next(c for t in proj["tracks"] for c in t["clips"] if c["id"] == r["clip"])
    assert copy["start_bar"] == 3 and [n["pitch"] for n in copy["notes"]] == [48, 55]

    params = (await s.ok("get_plugin_params", plugin_id=plugin))["params"]
    gain = next(p for p in params if p["name"] == "Gain")
    assert abs(gain["db"]) < 0.01
    await s.ok("set_plugin_params", plugin_id=plugin, values=[{"name": "Gain", "value": -6, "unit": "db"},
                                                            {"name": "Level", "value": 50, "unit": "percent"}])
    params = {p["name"]: p for p in (await s.ok("get_plugin_params", plugin_id=plugin))["params"]}
    assert abs(params["Gain"]["db"] + 6) < 0.05 and abs(params["Level"]["value"] - 0.5) < 1e-3

    await s.ok("set_tempo", bpm=128, at_bar=9)
    await s.ok("add_marker", name="Drop", bar=9)
    await s.ok("set_render_range", start_bar=1, end_bar=5)
    r = await s.ok("render_mix", out="mix.wav", stems=True)
    outs = [Path(o["path"]) for o in r["outputs"]]
    assert all(p.is_file() for p in outs) and len(outs) == 2
    assert abs(r["duration_sec"] - 4 * 4 * 60 / 124) < 0.01  # bars 1-5 at 124 bpm

    saved = await s.ok("save_project")
    assert saved["project"]["unsaved_changes"] is False
    res = await s.c.read_resource("flux://project")
    assert json.loads(res.contents[0].text)["title"] == "song"


async def test_one_tool_call_is_one_undo_step(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    track = (await s.ok("add_track", name="Keys", plugin="Test Synth"))["track"]
    before = session.describe(include_notes=True)
    # a multi-note clip and a 3-parameter change are each one step
    await s.ok("write_midi_clip", track_id=track, start_bar=1, length_bars=1,
               notes=[{"pitch": p, "start_beat": i, "length_beats": 1} for i, p in enumerate([60, 64, 67, 72])])
    r = await s.ok("undo_last_turn")
    assert r["undone"] == ["Agent: write clip on Keys, bars 1-2"]
    assert session.describe(include_notes=True) == before  # restored exactly
    r = await s.ok("redo")
    assert len(_clip_ids(session.describe())) == 1


async def test_failed_edit_changes_nothing(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    track = (await s.ok("add_track", name="Keys", plugin="Test Synth"))["track"]
    before = session.describe(include_notes=True)
    hist = session.history()
    # first step succeeds inside the turn, second fails: the whole turn must be discarded
    with pytest.raises(Exception):
        session.edit("two steps", [("write_notes", {"track": track, "start_bar": 1, "length_bars": 1, "notes": []}),
                                   ("write_notes", {"track": "nope", "start_bar": 1, "length_bars": 1, "notes": []})])
    assert session.describe(include_notes=True) == before
    assert session.history()["undo_text"] == hist["undo_text"]
    # and through the tools: a note that overruns the clip is rejected before reaching the engine
    msg = await s.err("write_midi_clip", track_id=track, start_bar=1, length_bars=1,
                      notes=[{"pitch": 60, "start_beat": 3, "length_beats": 2}])
    assert "ends at beat 5" in msg
    assert session.describe(include_notes=True) == before


async def test_dry_run_changes_nothing(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    before = session.describe()
    r = await s.ok("add_track", name="Ghost", plugin="Test Synth", dry_run=True)
    assert r["dry_run"] is True and r["changes"][0]["op"] == "add_track"
    r = await s.ok("set_tempo", bpm=90, dry_run=True)
    assert session.describe() == before


async def test_error_paths(flux):
    s, session, tmp = flux
    assert "new_project or open_project" in await s.err("add_track", name="X")
    await s.ok("new_project", dir="p")
    assert "get_project" in await s.err("write_midi_clip", track_id="nope", start_bar=1, length_bars=1, notes=[])
    assert "cannot read pitch" in await s.err("write_midi_clip", track_id="nope", start_bar=1, length_bars=1,
                                              notes=[{"pitch": "H2", "start_beat": 0, "length_beats": 1}])
    assert "not found" in await s.err("open_project", dir=str(tmp / "missing"))
    assert "file not found" in await s.err("import_file", path=str(tmp / "missing.wav"))

    # overwriting needs explicit consent
    (tmp / "p" / "out.wav").write_bytes(b"x")
    assert "refusing to overwrite" in await s.err("render_mix", out="out.wav")
    await s.ok("render_mix", out="out.wav", overwrite=True, start_bar=1, end_bar=2)

    # unsaved work is protected
    await s.ok("set_tempo", bpm=100)
    assert "unsaved edits" in await s.err("new_project", dir="q")
    await s.ok("new_project", dir="q", discard_unsaved=True)

    # undo with nothing to undo is a clear error
    msg = await s.err("undo_last_turn")
    assert "undo" in msg.lower()


async def test_root_restriction(engine_env, tmp_path):
    server = build_server(Session(Config(workdir=tmp_path, root=tmp_path / "jail")))
    (tmp_path / "jail").mkdir()
    async with Client(server) as c:
        r = await c.call_tool("new_project", {"dir": str(tmp_path / "outside")})
        assert r.is_error and "outside the allowed root" in r.content[0].text
        r = await c.call_tool("new_project", {"dir": str(tmp_path / "jail" / "song")})
        assert not r.is_error


async def test_engine_restart_after_crash(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    await s.ok("add_track", name="Keys", plugin="Test Synth")
    await s.ok("save_project")
    session.engine._proc.kill()  # simulate a crash between calls
    session.engine._proc.wait()
    r = await s.ok("get_project")
    assert "restarted" in r["notice"] and "Reopened" in r["notice"]
    assert any(t["name"] == "Keys" for t in r["tracks"])  # the saved project was reopened


async def test_generate_with_sine_backend_and_progress(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    seen = []

    async def progress(p, total, message):
        seen.append((p, total, message))

    r = await s.c.call_tool("generate_audio", {"prompt": "tone", "seconds": 2, "backend": "sine-test"},
                            progress_callback=progress)
    assert not r.is_error, r.content[0].text
    out = r.structured_content
    assert out["backend"] == "sine-test" and Path(out["provenance"]).is_file()
    assert seen and seen[-1][0] == seen[-1][1] == 100.0
    assert [p for p, _, _ in seen] == sorted(p for p, _, _ in seen)  # monotonic
    assert out["import"]["undo_step"].startswith("Agent: import generated")
    assert any(t["type"] == "audio" for t in out["import"]["project"]["tracks"])


async def test_licences_are_never_auto_accepted(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    backends = {b["name"]: b for b in (await s.ok("list_backends"))["backends"]}
    assert backends["ace-step"]["licence_accepted"] is False  # FLUX_AI_CACHE is a fresh temp dir
    msg = await s.err("generate_audio", prompt="house", backend="ace-step")
    assert "LicenceNotAccepted" in msg and "ace-step-1.5" in msg and "explicitly agree" in msg
    msg = await s.err("generate_audio", prompt="house", backend="ace-step", accept_licence="magenta-rt-2")
    assert "does not match" in msg
    # nothing was recorded as accepted
    backends = {b["name"]: b for b in (await s.ok("list_backends", refresh=True))["backends"]}
    assert not any(b["licence_accepted"] for b in backends.values() if b["licence_id"] != "none")


async def test_longform_reports_unavailable_or_runs(flux):
    s, session, tmp = flux
    await s.ok("new_project", dir="p")
    if not session.longform_available():
        msg = await s.err("plan_longform", genre="deep house", hours=1)
        assert "unavailable" in msg
    else:  # pragma: no cover - depends on the long-form orchestrator branch
        r = await s.ok("plan_longform", genre="deep house", hours=1)
        assert r["plan_path"]
