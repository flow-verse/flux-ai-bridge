# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool listing, schemas, annotations and unit conversions (no engine needed)."""
import pytest

from mcp import Client

from flux_mcp.music import db_to_gain, gain_to_db, pitch_to_midi, summarize
from flux_mcp.server import build_server

pytestmark = pytest.mark.anyio

READ_ONLY = {"get_project", "find_plugins", "get_plugin_params", "list_backends"}
DESTRUCTIVE = {"new_project", "open_project", "save_project", "render_mix", "undo_last_turn", "redo"}
EDITS = {"add_track", "load_plugin", "set_plugin_params", "write_midi_clip", "copy_clip", "import_file",
         "set_tempo", "add_marker", "set_render_range"}


async def _tools():
    async with Client(build_server()) as c:
        return {t.name: t for t in (await c.list_tools()).tools}


async def test_tool_set_is_curated():
    tools = await _tools()
    assert 15 <= len(tools) <= 25, sorted(tools)
    assert READ_ONLY | DESTRUCTIVE | EDITS <= set(tools)
    for t in tools.values():
        assert t.description and len(t.description) > 40, f"{t.name} needs a description that teaches its use"
        assert t.annotations is not None, f"{t.name} has no annotations"
        assert t.input_schema["type"] == "object"


async def test_annotations():
    tools = await _tools()
    for name in READ_ONLY:
        assert tools[name].annotations.read_only_hint is True, name
    for name in DESTRUCTIVE:
        a = tools[name].annotations
        assert a.read_only_hint is False and a.destructive_hint is True, name
    for name in EDITS:
        a = tools[name].annotations
        assert a.read_only_hint is False and a.destructive_hint is False, name
    a = tools["set_plugin_params"].annotations
    assert a.idempotent_hint is True and a.destructive_hint is False
    assert tools["generate_audio"].annotations.open_world_hint is True  # downloads model weights


async def test_schemas_carry_units_and_ranges():
    tools = await _tools()
    clip = tools["write_midi_clip"].input_schema
    note = clip["$defs"]["Note"]["properties"]
    assert note["velocity"]["minimum"] == 1 and note["velocity"]["maximum"] == 127
    assert note["start_beat"]["minimum"] == 0 and "exclusiveMinimum" in note["length_beats"]
    assert "C4" in note["pitch"]["description"]
    assert clip["properties"]["start_bar"]["minimum"] == 1
    tempo = tools["set_tempo"].input_schema["properties"]["bpm"]
    assert (tempo["minimum"], tempo["maximum"]) == (20, 999)
    unit = tools["set_plugin_params"].input_schema["$defs"]["ParamValue"]["properties"]["unit"]
    assert set(unit["enum"]) == {"normalized", "plain", "db", "percent"}
    gen = tools["generate_audio"].input_schema["properties"]
    assert "explicitly agreed" in gen["accept_licence"]["description"]
    assert "accept_licence" not in tools["generate_audio"].input_schema.get("required", [])
    for name in EDITS:
        assert "dry_run" in tools[name].input_schema["properties"], name


async def test_resources_listed():
    async with Client(build_server()) as c:
        uris = {str(r.uri) for r in (await c.list_resources()).resources}
    assert {"flux://project", "flux://plan", "flux://backends"} <= uris


async def test_plan_resource_without_plan():
    async with Client(build_server()) as c:
        r = await c.read_resource("flux://plan")
    assert '"plan": null' in r.contents[0].text


async def test_validation_happens_before_the_engine(monkeypatch):
    # an out-of-range bpm is rejected by the schema; no engine is needed (FLUX_ENGINE points nowhere)
    monkeypatch.setenv("FLUX_ENGINE", "/nonexistent/flux-engine")
    async with Client(build_server()) as c:
        r = await c.call_tool("set_tempo", {"bpm": 5000})
        assert r.is_error
        r = await c.call_tool("get_project", {})
        assert r.is_error and "flux-engine not found" in r.content[0].text



async def test_run_longform_renderer_choice(monkeypatch, tmp_path):
    import flux_mcp.server as server

    seen = []

    def fake_oneshot(args, flux_ai, progress, timeout):
        seen.append(list(args))
        return {"event": "done", "outputs": []}

    monkeypatch.setattr(server, "run_oneshot", fake_oneshot)
    mcp = build_server(server.Session(server.Config(engine="/opt/flux-engine", workdir=tmp_path)))
    mcp.session._longform = True
    async with Client(mcp) as c:
        schema = {t.name: t for t in (await c.list_tools()).tools}["run_longform"].input_schema["properties"]
        assert set(schema["renderer"]["enum"]) == {"builtin", "flux-host"} and schema["renderer"]["default"] == "builtin"
        r = await c.call_tool("run_longform", {"genre": "techno", "hours": 0.1})
        assert not r.is_error and "--renderer" not in seen[-1]
        r = await c.call_tool("run_longform", {"genre": "techno", "hours": 0.1, "renderer": "flux-host",
                                               "flux_plugins": {"default": "Test Synth", "drums": "Kit"}})
        assert not r.is_error
        a = seen[-1]
        assert a[a.index("--renderer") + 1] == "flux-host" and a[a.index("--flux-engine") + 1] == "/opt/flux-engine"
        assert {"default=Test Synth", "drums=Kit"} == {a[i + 1] for i, x in enumerate(a) if x == "--flux-plugin"}
        r = await c.call_tool("run_longform", {"genre": "techno", "hours": 0.1, "flux_plugins": {"bass": "X"}})
        assert r.is_error and "flux-host" in r.content[0].text

def test_pitch_names():
    assert pitch_to_midi("C4") == 60
    assert pitch_to_midi("A4") == 69
    assert pitch_to_midi("C#4") == pitch_to_midi("Db4") == 61
    assert pitch_to_midi("Bb1") == 34
    assert pitch_to_midi("C-1") == 0
    assert pitch_to_midi(127) == 127 and pitch_to_midi("64") == 64
    for bad in ("H2", "C", 128, -1, 60.5, "G9#"):
        with pytest.raises(ValueError):
            pitch_to_midi(bad)


def test_db_conversion():
    assert abs(db_to_gain(-6) - 0.501187) < 1e-5
    assert abs(gain_to_db(0.5) + 6.0206) < 1e-3
    assert gain_to_db(0) is None


def test_summary_hides_system_tracks():
    proj = {"title": "t", "dir": "/x", "tempo": {"bpm": 120, "changes": []},
            "time_signature": {"numerator": 4, "denominator": 4, "changes": []}, "range": {}, "markers": [],
            "tracks": [{"id": "1", "name": "Chords", "type": "chord", "plugins": [], "clips": []},
                       {"id": "2", "name": "Bass", "type": "instrument", "plugins": [], "clips": []}]}
    s = summarize(proj)
    assert [t["name"] for t in s["tracks"]] == ["Bass"]
    assert summarize(None) == {"open": False}
