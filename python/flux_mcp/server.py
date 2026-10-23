# SPDX-License-Identifier: GPL-3.0-or-later
"""The flux-mcp MCP server: a curated tool set over flux-engine and flux-ai.

Design rules (see docs/mcp.md and the DAWZY lessons in the P3 report):
- a small set of task-shaped tools, not a 1:1 dump of the engine API;
- every edit tool runs inside one engine agent turn, so one ``undo_last_turn``
  reverts the whole tool call; if any step fails the turn is discarded, so a
  failed tool call leaves the project unchanged;
- every edit tool returns the fresh project summary, so the model never acts on
  stale state;
- unit conversions (note names, dB) happen in the tools, not in the model;
- tools that discard or overwrite are annotated ``destructiveHint`` and refuse to
  overwrite unless asked explicitly; model licences are never accepted implicitly.
"""
from __future__ import annotations

import functools
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from flux_mcp import __version__
from flux_mcp.engine import (NO_PROJECT, STATE_CONFLICT, EngineClient, EngineCrashed, EngineError,
                             EngineNotFound)
from flux_mcp.music import db_to_gain, gain_to_db, pitch_to_midi, summarize
from flux_mcp.worker import AiWorker, JobError, WorkerUnavailable, has_subcommand, run_oneshot

INSTRUCTIONS = """\
Flux is a local, open-source DAW. These tools drive a headless Flux engine.

How to work reliably:
1. Call get_project first, and again whenever you are unsure of the state. Every edit tool also returns
   the fresh project summary; use the ids in it. Tracks, clips and plugins are addressed by id, never by
   index or position.
2. Time is musical: bars are 1-based (bar 1 is the start of the song) and range ends are exclusive
   (bars 1 to 9 is eight bars). Note positions inside a clip are beats counted from the clip start
   (0-based; a beat is a quarter note in 4/4).
3. Pitches may be MIDI numbers or note names (C4 = 60, middle C).
4. Each edit tool call is one undo step ("Agent: ..."). If it fails, nothing is changed. undo_last_turn
   reverts the previous tool call. There is no delete or in-place edit yet: to change a clip, write a new
   one (copy_clip can transpose).
5. Nothing is saved until save_project. new_project and open_project refuse to drop unsaved edits unless
   discard_unsaved is true, which needs the user's agreement.
6. AI generation downloads model weights under their own licences. Never set accept_licence unless the
   user has explicitly agreed to that licence in this conversation.
"""

NOTE_DESC = ("One note. pitch: MIDI number 0-127 or a name like 'C4' (C4 = 60, middle C), 'F#2', 'Bb3'. "
             "start_beat: beats from the clip start, 0-based (bar 2 beat 1 of a 4/4 clip is 4.0). "
             "length_beats: duration in beats (> 0); the note must end inside the clip. velocity: 1-127.")


class Note(BaseModel):
    """A MIDI note in musical time, relative to its clip."""
    pitch: Annotated[int | str, Field(description="MIDI number 0-127 or note name such as 'C4' (= 60), 'F#2', 'Bb3'")]
    start_beat: Annotated[float, Field(ge=0, description="Beats from the clip start, 0-based (a quarter note in 4/4)")]
    length_beats: Annotated[float, Field(gt=0, le=4096, description="Duration in beats, > 0")]
    velocity: Annotated[int, Field(ge=1, le=127, description="MIDI velocity 1-127")] = 100


class ParamValue(BaseModel):
    """One plugin parameter assignment."""
    name: Annotated[str | None, Field(description="Parameter name as listed by get_plugin_params (case-insensitive)")] = None
    id: Annotated[str | None, Field(description="Stable parameter id from get_plugin_params (use instead of name)")] = None
    value: Annotated[float | str | bool, Field(description="The value, in `unit`. Enum parameters also accept their label; toggles accept 'on'/'off'/true/false")]
    unit: Annotated[Literal["normalized", "plain", "db", "percent"], Field(
        description="normalized: 0-1 of the parameter's range (default). plain: the parameter's own units, within "
                    "min..max. db: decibels, only for type 'gain' parameters (0 dB = unity; converted for you). "
                    "percent: 0-100 of the range.")] = "normalized"


@dataclass
class Config:
    engine: str | None = None
    engine_args: list[str] = field(default_factory=list)
    flux_ai: list[str] | None = None
    root: Path | None = None              # if set, every file path must be inside it
    workdir: Path = field(default_factory=Path.cwd)
    call_timeout: float = 60.0
    render_timeout: float = 1800.0
    job_timeout: float = 3 * 3600.0


class Session:
    """State shared by all tools: the engine, the AI worker and small caches."""

    def __init__(self, config: Config | None = None, engine: EngineClient | None = None, worker: AiWorker | None = None):
        self.config = config or Config()
        self.engine = engine or EngineClient(self.config.engine, self.config.engine_args)
        self.engine.DEFAULT_TIMEOUT = self.config.call_timeout
        self.worker = worker or AiWorker(self.config.flux_ai)
        self.saved_index: int | None = None   # undo index at the last create/open/save
        self.latest_plan: dict | None = None
        self.latest_plan_path: str | None = None
        self.backends_cache: list[dict] | None = None
        self._longform: bool | None = None

    def close(self) -> None:
        self.engine.close()
        self.worker.close()

    # -- paths ----------------------------------------------------------------
    def path(self, p: str, what: str = "path") -> Path:
        q = Path(os.path.expanduser(p))
        if not q.is_absolute():
            q = self.config.workdir / q
        q = Path(os.path.normpath(q))
        root = self.config.root
        if root is not None:
            r = Path(os.path.normpath(root.resolve()))
            try:
                q.resolve().relative_to(r)
            except ValueError:
                raise ToolError(f"{what} '{p}' is outside the allowed root {r} (flux-mcp --root)")
        return q

    # -- engine helpers ---------------------------------------------------------
    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> Any:
        return self.engine.call(method, params or {}, timeout)

    def describe(self, include_notes: bool = False) -> dict | None:
        try:
            return self.call("describe_project", {"include_notes": include_notes})
        except EngineError as e:
            if e.code == NO_PROJECT:
                return None
            raise

    def history(self) -> dict | None:
        try:
            return self.call("get_history")
        except EngineError:
            return None

    def summary(self) -> dict[str, Any]:
        proj = self.describe()
        out = summarize(proj, self.history() if proj else None)
        if proj is not None:
            out["unsaved_changes"] = self.dirty()
        return out

    def dirty(self) -> bool:
        h = self.history()
        if h is None:
            return False
        return self.saved_index is None or h.get("index") != self.saved_index

    def mark_saved(self) -> None:
        h = self.history()
        self.saved_index = h.get("index") if h else None

    def edit(self, label: str, steps: list[tuple[str, dict]], dry_run: bool = False) -> dict[str, Any]:
        """Run engine edits as one agent turn; discard the turn if any step fails."""
        if dry_run:
            results = [self.call(m, {**p, "dry_run": True}) for m, p in steps]
            changes = [c for r in results if isinstance(r, dict) for c in r.get("changes", [])]
            return {"dry_run": True, "changes": changes, "results": results}
        try:
            self.call("begin_agent_turn", {"label": label})
        except EngineError as e:
            if e.code != STATE_CONFLICT:
                raise
            # a turn left open (e.g. by an earlier crash of this adapter): close it, then start ours
            self.call("end_agent_turn", {})
            self.call("begin_agent_turn", {"label": label})
        results = []
        try:
            for m, p in steps:
                results.append(self.call(m, p))
        except EngineError:
            try:
                self.call("end_agent_turn", {"discard": True})
            except Exception:
                pass
            raise
        end = self.call("end_agent_turn", {})
        changes = [c for r in results if isinstance(r, dict) for c in r.get("changes", [])]
        return {"undo_step": f"Agent: {label}", "recorded": end.get("recorded", True), "changes": changes,
                "results": results, "project": self.summary()}

    # -- flux-ai helpers ----------------------------------------------------------
    def backends(self, refresh: bool = False) -> list[dict]:
        if self.backends_cache is None or refresh:
            done = self.worker.run("backends", {}, timeout=300)
            self.backends_cache = done.get("backends", [])
        return self.backends_cache

    def longform_available(self) -> bool:
        if self._longform is None:
            self._longform = has_subcommand("longform", self.config.flux_ai)
        return self._longform

    def default_out_dir(self, sub: str) -> Path:
        proj = None
        try:
            proj = self.describe()
        except Exception:
            pass
        base = Path(proj["dir"]) if proj and proj.get("dir") else self.config.workdir
        d = base / sub
        return d


def _error_text(e: Exception) -> str:
    if isinstance(e, EngineError):
        hint = {
            NO_PROJECT: " Call new_project or open_project first.",
            -32001: " Call get_project for current ids.",
            STATE_CONFLICT: "",
        }.get(e.code, "")
        return f"{e.message}.{hint}" if not e.message.endswith(".") else f"{e.message}{hint}"
    if isinstance(e, (EngineCrashed, EngineNotFound, WorkerUnavailable)):
        return str(e)
    if isinstance(e, JobError):
        return f"{e} ({e.error_type})" if e.error_type else str(e)
    if isinstance(e, (ValueError, KeyError, FileNotFoundError, FileExistsError)):
        return str(e).strip("'\"")
    return f"{type(e).__name__}: {e}"


def _tool_errors(session: Session):
    """Turn every failure into a ToolError with a readable message (the SDK hides other exceptions)."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                result = await fn(*args, **kwargs)
            except ToolError:
                raise
            except Exception as e:
                msg = _error_text(e)
                if session.engine.notice:
                    msg = session.engine.notice + " " + msg
                    session.engine.notice = None
                raise ToolError(msg) from e
            if session.engine.notice and isinstance(result, dict):
                result = {"notice": session.engine.notice, **result}
                session.engine.notice = None
            return result
        return wrapper
    return deco


async def _run(fn, *args):
    return await anyio.to_thread.run_sync(functools.partial(fn, *args))


async def _with_heartbeat(ctx: Context | None, fn, message: str):
    """Run a blocking call in a thread, reporting elapsed seconds as progress once a second."""
    result: dict = {}

    async def work():
        result["v"] = await anyio.to_thread.run_sync(fn)
        tg.cancel_scope.cancel()

    async def beat():
        t0 = time.monotonic()
        while True:
            await anyio.sleep(1.0)
            if ctx is not None:
                s = time.monotonic() - t0
                try:
                    await ctx.report_progress(round(s, 1), None, f"{message} ({s:.0f} s)")
                except Exception:
                    return

    async with anyio.create_task_group() as tg:
        tg.start_soon(beat)
        tg.start_soon(work)
    return result["v"]


def _progress_bridge(ctx: Context | None, scale: float = 100.0):
    """A thread-side progress callback that forwards flux-ai fractions as MCP progress (0-100)."""
    if ctx is None:
        return None

    def cb(fraction: float, message: str) -> None:
        try:
            anyio.from_thread.run(ctx.report_progress, round(fraction * scale, 2), scale, message or None)
        except Exception:
            pass
    return cb


RO = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
ADD = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
SET = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)
OVERWRITE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)
AI = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)

BarF = Annotated[float, Field(ge=1, le=100000, description="Bar position, 1-based; may be fractional (2.5 = half-way through bar 2)")]
Bpm = Annotated[float, Field(ge=20, le=999, description="Tempo in beats per minute, 20-999")]
Id = Annotated[str, Field(min_length=1, description="An id (UUID) from get_project or from the tool that created it")]


def build_server(session: Session | None = None) -> MCPServer:
    s = session or Session()

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield s
        finally:
            await anyio.to_thread.run_sync(s.close)

    mcp = MCPServer("flux", title="Flux", description="Compose, arrange and render music in the Flux DAW",
                    instructions=INSTRUCTIONS, version=__version__, lifespan=lifespan)
    guard = _tool_errors(s)

    # ------------------------------------------------------------------ read
    @mcp.tool(annotations=RO)
    @guard
    async def get_project(include_notes: Annotated[bool, Field(description="Also list every note of every MIDI clip (larger output)")] = False) -> dict[str, Any]:
        """Read the open project: tempo, time signature, render range, markers, tracks with their ids,
        plugins and clips, plus undo history and whether there are unsaved changes. Call this before
        editing and whenever you need ids. Returns {"open": false} when no project is open."""
        def go():
            proj = s.describe(include_notes)
            if proj is None:
                return {"open": False, "hint": "No project is open: call new_project or open_project."}
            return {"open": True, **proj, "history": s.history(), "unsaved_changes": s.dirty()}
        return await _run(go)

    @mcp.tool(annotations=RO)
    @guard
    async def find_plugins(
        query: Annotated[str | None, Field(description="Case-insensitive match on name, vendor or category")] = None,
        instruments_only: Annotated[bool, Field(description="Only instruments (synths, samplers)")] = False,
        format: Annotated[Literal["clap", "vst3"] | None, Field(description="Restrict to one plugin format")] = None,
        rescan: Annotated[bool, Field(description="Scan the plugin folders again (slow)")] = False,
    ) -> dict[str, Any]:
        """Search installed plugins. Use the returned `id` (or exact `name`) as `plugin` in add_track or load_plugin."""
        params: dict = {"instruments_only": instruments_only, "rescan": rescan}
        if query:
            params["query"] = query
        if format:
            params["format"] = format
        return await _run(lambda: s.call("list_plugins", params, 300))

    @mcp.tool(annotations=RO)
    @guard
    async def get_plugin_params(
        plugin_id: Annotated[Id, Field(description="Plugin instance id from get_project (tracks[].plugins[].id)")],
        include_hidden: bool = False,
    ) -> dict[str, Any]:
        """List a plugin instance's parameters with id, name, type, current value (normalized 0-1 and plain),
        min/max in plain units, and, for 'gain' parameters, the current level in dB. Every plugin also has
        the host's own 'Bypass' and 'Gain' parameters."""
        def go():
            r = s.call("list_params", {"plugin": plugin_id, "include_hidden": include_hidden})
            for p in r.get("params", []):
                if p.get("type") == "gain" and isinstance(p.get("plain_value"), (int, float)):
                    db = gain_to_db(p["plain_value"])
                    p["db"] = None if db is None else round(db, 2)
            return r
        return await _run(go)

    @mcp.tool(annotations=RO)
    @guard
    async def list_backends(refresh: Annotated[bool, Field(description="Re-check availability instead of using the cached list")] = False) -> dict[str, Any]:
        """List the AI generation backends: availability (and why not), modes, model tiers, output kind
        (audio or midi), the weights licence id, URL and commercial-use terms, and whether that licence has
        been accepted. Use before generate_audio."""
        return {"backends": await _run(s.backends, refresh)}

    # --------------------------------------------------------------- project
    def _check_unsaved(discard_unsaved: bool) -> None:
        if not discard_unsaved and s.describe() is not None and s.dirty():
            raise ToolError("The open project has unsaved edits. Call save_project first, or pass "
                            "discard_unsaved=true only if the user agreed to lose them.")

    @mcp.tool(annotations=DESTRUCTIVE)
    @guard
    async def new_project(
        dir: Annotated[str, Field(description="Directory for the new project; must not exist or be empty. Relative paths are relative to the server's working directory")],
        bpm: Bpm = 120.0,
        time_signature: Annotated[str, Field(pattern=r"^\d{1,2}/(1|2|4|8|16|32)$", description="e.g. '4/4', '6/8'")] = "4/4",
        discard_unsaved: Annotated[bool, Field(description="Close the current project even if it has unsaved edits (needs the user's agreement)")] = False,
    ) -> dict[str, Any]:
        """Create a new, empty project (saved immediately) and make it the open project. The project title is
        the directory name. Refuses if the open project has unsaved edits, unless discard_unsaved is true."""
        def go():
            _check_unsaved(discard_unsaved)
            d = s.path(dir, "dir")
            r = s.call("create_project", {"dir": str(d), "bpm": bpm, "time_signature": time_signature}, 120)
            s.mark_saved()
            return {"created": r, "project": s.summary()}
        return await _run(go)

    @mcp.tool(annotations=DESTRUCTIVE)
    @guard
    async def open_project(
        dir: Annotated[str, Field(description="Directory of a saved Flux project")],
        discard_unsaved: Annotated[bool, Field(description="Close the current project even if it has unsaved edits (needs the user's agreement)")] = False,
    ) -> dict[str, Any]:
        """Open a saved project (closing the current one). Refuses if the open project has unsaved edits,
        unless discard_unsaved is true."""
        def go():
            _check_unsaved(discard_unsaved)
            d = s.path(dir, "dir")
            if not d.is_dir():
                raise ToolError(f"project directory not found: {d}")
            s.call("open_project", {"dir": str(d)}, 300)
            s.mark_saved()
            return {"project": s.summary()}
        return await _run(go)

    @mcp.tool(annotations=OVERWRITE)
    @guard
    async def save_project(
        dir: Annotated[str | None, Field(description="Save as: a new directory (must not exist or be empty unless overwrite is true). Omit to save in place")] = None,
        overwrite: Annotated[bool, Field(description="Allow 'save as' into a non-empty directory")] = False,
    ) -> dict[str, Any]:
        """Save the open project, in place or to a new directory ("save as", which becomes the current
        directory). Saving in place overwrites the previous saved version."""
        def go():
            params = {}
            if dir:
                d = s.path(dir, "dir")
                if d.exists() and any(d.iterdir()) and not overwrite:
                    cur = (s.describe() or {}).get("dir")
                    if cur is None or Path(cur) != d:
                        raise ToolError(f"{d} is not empty; pass overwrite=true only if the user agreed to replace it")
                params["dir"] = str(d)
            r = s.call("save_project", params, 300)
            s.mark_saved()
            return {"saved": r, "project": s.summary()}
        return await _run(go)

    # ------------------------------------------------------------------ edits
    @mcp.tool(annotations=ADD)
    @guard
    async def add_track(
        kind: Annotated[Literal["instrument", "midi", "audio"], Field(description="instrument: MIDI clips played by an instrument plugin (most common). audio: audio clips. midi: MIDI only, no instrument")] = "instrument",
        name: Annotated[str | None, Field(max_length=100, description="Track name (made unique if needed), e.g. 'Drums', 'Bass', 'Pad'")] = None,
        plugin: Annotated[str | None, Field(description="Plugin id or exact name from find_plugins: an instrument for instrument tracks, an effect otherwise")] = None,
        dry_run: Annotated[bool, Field(description="Validate and report the changes without applying them")] = False,
    ) -> dict[str, Any]:
        """Add a track (optionally with a plugin) as one undo step. Returns the new track id and the fresh
        project summary. An instrument track needs an instrument plugin to make sound."""
        params: dict = {"type": kind}
        if name:
            params["name"] = name
        if plugin:
            params["plugin_id"] = plugin
        label = f"add {kind} track" + (f" {name}" if name else "")
        r = await _run(s.edit, label, [("add_track", params)], dry_run)
        if not dry_run:
            r["track"] = r["results"][0].get("track")
        return r

    @mcp.tool(annotations=ADD)
    @guard
    async def load_plugin(
        track_id: Id,
        plugin: Annotated[str, Field(description="Plugin id or exact name from find_plugins")],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Add a plugin to an existing track: an instrument to an instrument track that has none, otherwise an
        effect after the existing ones. One undo step."""
        return await _run(s.edit, f"load {plugin}", [("load_plugin", {"track": track_id, "plugin_id": plugin})], dry_run)

    @mcp.tool(annotations=SET)
    @guard
    async def set_plugin_params(
        plugin_id: Annotated[Id, Field(description="Plugin instance id from get_project")],
        values: Annotated[list[ParamValue], Field(min_length=1, max_length=64, description="Parameters to set; all are validated before any is applied")],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Set one or more plugin parameters as one undo step. Prefer unit 'plain' with the min/max from
        get_plugin_params, or 'db' for gain parameters (e.g. the host 'Gain': -6 dB halves the level).
        Setting the same values again changes nothing."""
        def go():
            needs_lookup = any(v.unit in ("db", "percent") for v in values)
            params_by = {}
            if needs_lookup:
                listed = s.call("list_params", {"plugin": plugin_id, "include_hidden": True}).get("params", [])
                for p in listed:
                    params_by[("id", str(p["id"]))] = p
                    params_by[("name", str(p["name"]).lower())] = p
            out = []
            for v in values:
                if not v.name and not v.id:
                    raise ToolError("each value needs a parameter 'name' or 'id'")
                entry: dict = {"id": v.id} if v.id else {"name": v.name}
                if v.unit in ("db", "percent"):
                    p = params_by.get(("id", v.id)) if v.id else params_by.get(("name", (v.name or "").lower()))
                    if p is None:
                        raise ToolError(f"unknown parameter {v.id or v.name!r}; call get_plugin_params")
                    try:
                        num = float(v.value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        raise ToolError(f"{v.unit} values must be numbers, got {v.value!r}")
                    if v.unit == "db":
                        if p.get("type") != "gain":
                            raise ToolError(f"parameter {p['name']!r} is type {p.get('type')!r}; unit 'db' only works for "
                                            "'gain' parameters (use 'plain' or 'normalized')")
                        g = db_to_gain(num)
                        if not p["min"] <= g <= p["max"] + 1e-9:
                            lo = gain_to_db(p["min"]) if p["min"] > 0 else None
                            hi = gain_to_db(p["max"])
                            raise ToolError(f"{num} dB is outside the range of {p['name']!r} "
                                            f"({'-inf' if lo is None else round(lo, 1)} to {round(hi or 0, 1)} dB)")
                        entry.update(value=g, unit="plain")
                    else:
                        if not 0 <= num <= 100:
                            raise ToolError("percent values must be 0-100")
                        entry.update(value=num / 100.0, unit="normalized")
                else:
                    entry.update(value=v.value, unit=v.unit)
                out.append(entry)
            names = ", ".join(str(e.get("name") or e.get("id")) for e in out)
            return s.edit(f"set {names}", [("set_params", {"plugin": plugin_id, "values": out})], dry_run)
        return await _run(go)

    @mcp.tool(annotations=ADD)
    @guard
    async def write_midi_clip(
        track_id: Annotated[Id, Field(description="An instrument or MIDI track id")],
        start_bar: BarF,
        length_bars: Annotated[float, Field(gt=0, le=10000, description="Clip length in bars")],
        notes: Annotated[list[Note], Field(max_length=20000, description=NOTE_DESC)],
        name: Annotated[str | None, Field(max_length=100, description="Clip name")] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Create one MIDI clip of length_bars bars at start_bar and fill it with notes, as one undo step.
        Note times are beats from the clip start (0-based); in 4/4 a 4-bar clip spans beats 0-16 and bar 3
        starts at beat 8. Notes that would end after the clip are rejected (never truncated). An empty
        list makes an empty clip. Returns the clip id and the fresh project summary."""
        def go():
            proj = s.describe()
            if proj is None:
                raise EngineError(NO_PROJECT, "No project is open")
            num = (proj.get("time_signature") or {}).get("numerator", 4)
            clip_beats = length_bars * num
            out = []
            for i, n in enumerate(notes):
                try:
                    pitch = pitch_to_midi(n.pitch)
                except ValueError as e:
                    raise ToolError(f"notes[{i}]: {e}")
                if n.start_beat + n.length_beats > clip_beats + 1e-9:
                    raise ToolError(f"notes[{i}] ends at beat {n.start_beat + n.length_beats:g} but the clip is "
                                    f"{clip_beats:g} beats long ({length_bars:g} bars of {num}); shorten the note or "
                                    "lengthen the clip")
                out.append({"pitch": pitch, "start_beat": n.start_beat, "length_beats": n.length_beats, "velocity": n.velocity})
            params = {"track": track_id, "start_bar": start_bar, "length_bars": length_bars, "notes": out}
            if name:
                params["name"] = name
            tname = next((t["name"] for t in proj.get("tracks", []) if t["id"] == track_id), track_id[:8])
            end = start_bar + length_bars
            r = s.edit(f"write clip on {tname}, bars {start_bar:g}-{end:g}", [("write_notes", params)], dry_run)
            if not dry_run:
                r["clip"] = r["results"][0].get("clip")
            return r
        return await _run(go)

    @mcp.tool(annotations=ADD)
    @guard
    async def copy_clip(
        clip_id: Annotated[Id, Field(description="A MIDI clip id from get_project")],
        to_bar: Annotated[BarF, Field(description="Start bar of the copy")],
        transpose_semitones: Annotated[int, Field(ge=-48, le=48, description="Shift every note by this many semitones (12 = one octave up)")] = 0,
        to_track_id: Annotated[str | None, Field(description="Target track id (default: the clip's own track)")] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Copy a MIDI clip to another position or track, optionally transposed, as one undo step. Use this to
        repeat a pattern or to make a transposed variation (the original clip is kept; there is no
        in-place edit or delete yet)."""
        def go():
            proj = s.describe(include_notes=True)
            if proj is None:
                raise EngineError(NO_PROJECT, "No project is open")
            for t in proj.get("tracks", []):
                for c in t.get("clips", []):
                    if c["id"] == clip_id:
                        if c.get("kind") != "midi":
                            raise ToolError(f"clip {clip_id} is a {c.get('kind')} clip; only MIDI clips can be copied")
                        num = (proj.get("time_signature") or {}).get("numerator", 4)
                        notes = []
                        for n in c.get("notes", []):
                            p = n["pitch"] + transpose_semitones
                            if not 0 <= p <= 127:
                                raise ToolError(f"transposing pitch {n['pitch']} by {transpose_semitones} leaves 0-127")
                            notes.append({**n, "pitch": p})
                        params = {"track": to_track_id or t["id"], "start_bar": to_bar,
                                  "length_bars": c["length_beats"] / num, "notes": notes, "name": c.get("name") or t["name"]}
                        label = f"copy clip to bar {to_bar:g}" + (f", {transpose_semitones:+d} st" if transpose_semitones else "")
                        r = s.edit(label, [("write_notes", params)], dry_run)
                        if not dry_run:
                            r["clip"] = r["results"][0].get("clip")
                        return r
            raise EngineError(-32001, f"'{clip_id}' is not a clip id in the open project")
        return await _run(go)

    @mcp.tool(annotations=ADD)
    @guard
    async def import_file(
        path: Annotated[str, Field(description="Audio (.wav, .flac, .mp3, .ogg, .aiff) or MIDI (.mid, .midi) file")],
        start_bar: BarF = 1.0,
        track_id: Annotated[str | None, Field(description="Existing track to import into (audio track for audio, instrument/MIDI track for MIDI). Default: a new track named after the file")] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Import an audio or MIDI file into the arrangement at start_bar, as one undo step."""
        def go():
            p = s.path(path)
            if not p.is_file():
                raise ToolError(f"file not found: {p}")
            method = "import_midi" if p.suffix.lower() in (".mid", ".midi") else "import_audio"
            params: dict = {"path": str(p), "start_bar": start_bar}
            if track_id:
                params["track"] = track_id
            return s.edit(f"import {p.name} at bar {start_bar:g}", [(method, params)], dry_run)
        return await _run(go)

    @mcp.tool(annotations=SET)
    @guard
    async def set_tempo(bpm: Bpm, at_bar: Annotated[BarF, Field(description="Bar from which the tempo applies (1 = whole song)")] = 1.0,
                        dry_run: bool = False) -> dict[str, Any]:
        """Set the tempo from at_bar onwards (changes the tempo event at that bar, or adds one). One undo step."""
        return await _run(s.edit, f"tempo {bpm:g} bpm at bar {at_bar:g}", [("set_tempo", {"bpm": bpm, "at_bar": at_bar})], dry_run)

    @mcp.tool(annotations=ADD)
    @guard
    async def add_marker(name: Annotated[str, Field(min_length=1, max_length=100, description="e.g. 'Intro', 'Drop', 'Outro'")],
                         bar: BarF, dry_run: bool = False) -> dict[str, Any]:
        """Add a named arrangement marker at a bar. One undo step."""
        return await _run(s.edit, f"marker {name} at bar {bar:g}", [("add_marker", {"name": name, "bar": bar})], dry_run)

    @mcp.tool(annotations=SET)
    @guard
    async def set_render_range(start_bar: BarF, end_bar: Annotated[float, Field(gt=1, le=100001, description="Exclusive end bar (1 to 9 = eight bars)")],
                               dry_run: bool = False) -> dict[str, Any]:
        """Set the song's start and end markers, which render_mix uses when no range is given. One undo step."""
        if end_bar <= start_bar:
            raise ToolError("end_bar must be after start_bar")
        return await _run(s.edit, f"range bars {start_bar:g}-{end_bar:g}", [("set_range", {"start_bar": start_bar, "end_bar": end_bar})], dry_run)

    # ----------------------------------------------------------------- render
    @mcp.tool(annotations=OVERWRITE)
    @guard
    async def render_mix(
        out: Annotated[str, Field(description="Output file (.wav or .flac). Relative paths are relative to the project directory")],
        start_bar: Annotated[float | None, Field(ge=1, description="First bar (default: song start marker)")] = None,
        end_bar: Annotated[float | None, Field(gt=1, description="Exclusive end bar (default: song end marker); bars 1-9 = eight bars")] = None,
        format: Annotated[Literal["wav", "flac"] | None, Field(description="Default: from the file extension")] = None,
        bits: Annotated[Literal["16", "24", "32f"], Field(description="Bit depth; 32f (float) is WAV only")] = "24",
        stems: Annotated[bool, Field(description="Also write one file per track next to the mix ('<name> - NN <track>.<ext>')")] = False,
        overwrite: Annotated[bool, Field(description="Replace existing files (only with the user's agreement)")] = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Render the project offline to an audio file (and optional per-track stems). Not an edit and not
        undoable; it writes files to disk and refuses to replace existing files unless overwrite is true.
        Reports progress (elapsed seconds) while rendering."""
        def prepare():
            proj = s.describe()
            if proj is None:
                raise EngineError(NO_PROJECT, "No project is open")
            p = Path(os.path.expanduser(out))
            if not p.is_absolute() and proj.get("dir"):
                p = Path(proj["dir"]) / p
            p = s.path(str(p), "out")
            fmt = format
            if p.suffix.lower() not in (".wav", ".flac"):
                p = p.with_suffix(p.suffix + "." + (fmt or "wav")) if p.suffix else p.with_suffix("." + (fmt or "wav"))
            fmt = fmt or p.suffix.lower().lstrip(".")
            if fmt == "flac" and bits == "32f":
                raise ToolError("FLAC cannot store 32-bit float; use bits 16 or 24")
            existing = [p] if p.exists() else []
            if stems:
                existing += sorted(p.parent.glob(f"{p.stem} - *{p.suffix}"))
            if existing and not overwrite:
                raise ToolError("refusing to overwrite existing file(s): " + ", ".join(str(e) for e in existing[:5]) +
                                ". Choose another name, or pass overwrite=true if the user agreed.")
            if start_bar is not None and end_bar is not None and end_bar <= start_bar:
                raise ToolError("end_bar must be after start_bar")
            p.parent.mkdir(parents=True, exist_ok=True)
            params: dict = {"out": str(p), "format": fmt, "bits": bits if bits == "32f" else int(bits), "stems": stems}
            if start_bar is not None:
                params["start_bar"] = start_bar
            if end_bar is not None:
                params["end_bar"] = end_bar
            return params

        params = await _run(prepare)
        t0 = time.monotonic()
        if ctx is not None:
            await ctx.report_progress(0, None, "rendering")
        # flux-engine's render reports no progress of its own: send elapsed seconds as a heartbeat
        result = await _with_heartbeat(ctx, lambda: s.call("render", params, s.config.render_timeout), "rendering")
        if ctx is not None:
            el = round(time.monotonic() - t0, 2) + 0.01
            await ctx.report_progress(el, el, "done")
        return result

    # ----------------------------------------------------------------- history
    @mcp.tool(annotations=DESTRUCTIVE)
    @guard
    async def undo_last_turn() -> dict[str, Any]:
        """Undo the most recent undo step: the whole previous edit tool call (one agent turn) or one manual
        edit. Use redo to bring it back. Returns the fresh project summary."""
        def go():
            r = s.call("undo")
            return {"undone": [c.get("text") for c in r.get("changes", [])], "project": s.summary()}
        return await _run(go)

    @mcp.tool(annotations=DESTRUCTIVE)
    @guard
    async def redo() -> dict[str, Any]:
        """Redo the step most recently undone. Returns the fresh project summary."""
        def go():
            r = s.call("redo")
            return {"redone": [c.get("text") for c in r.get("changes", [])], "project": s.summary()}
        return await _run(go)

    # -------------------------------------------------------------------- AI
    def _new_out(sub: str, stem: str, ext: str, out: str | None) -> Path:
        if out:
            p = s.path(out, "out")
        else:
            p = s.default_out_dir(sub) / f"{stem}-{uuid.uuid4().hex[:8]}{ext}"
        if p.exists():
            raise ToolError(f"{p} already exists; choose another output path")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _import_outputs(label: str, files: list[str], start_bar: float, track_id: str | None) -> dict[str, Any]:
        steps = []
        for f in files:
            method = "import_midi" if f.lower().endswith((".mid", ".midi")) else "import_audio"
            params: dict = {"path": f, "start_bar": start_bar}
            if track_id:
                params["track"] = track_id
            steps.append((method, params))
        return s.edit(label, steps)

    @mcp.tool(annotations=AI)
    @guard
    async def generate_audio(
        prompt: Annotated[str, Field(min_length=1, max_length=2000, description="Text description: genre, instruments, mood")],
        seconds: Annotated[float, Field(gt=0, le=600, description="Length in seconds")] = 30.0,
        backend: Annotated[str, Field(description="Backend name from list_backends, e.g. 'ace-step', 'magenta-rt', 'stable-audio-3', 'amt' (MIDI), 'sine-test' (test tone)")] = "ace-step",
        bpm: Annotated[float | None, Field(ge=20, le=999, description="Tempo conditioning (default: none)")] = None,
        key: Annotated[str | None, Field(max_length=20, description="Key conditioning, e.g. 'A minor', 'F# major'")] = None,
        seed: Annotated[int, Field(ge=0, description="Random seed; the same seed and inputs reproduce the output")] = 0,
        model: Annotated[str | None, Field(description="Backend model tier, e.g. 'turbo' (ace-step), 'base'/'small' (magenta-rt)")] = None,
        out: Annotated[str | None, Field(description="Output path (default: <project>/ai/gen-<id>.wav); must not exist")] = None,
        accept_licence: Annotated[str | None, Field(description="The backend's licence id (from list_backends). Set ONLY after the user has read and explicitly agreed to that licence in this conversation; never on your own")] = None,
        import_to_project: Annotated[bool, Field(description="Import the result into the open project as one undo step")] = True,
        start_bar: Annotated[BarF, Field(description="Where to import the result")] = 1.0,
        track_id: Annotated[str | None, Field(description="Import into this existing track (default: a new track)")] = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Generate audio (or MIDI, for the 'amt' backend) with a local AI model via the flux-ai warm worker,
        then optionally import it into the project. Weights download on first use under the model's own
        licence: if it has not been accepted the tool fails with LicenceNotAccepted and names the licence;
        show that to the user and only retry with accept_licence after they explicitly agree. Every output
        gets a provenance sidecar (<out>.provenance.json) marking it AI-generated. Reports progress."""
        def prep():
            known = {b["name"]: b for b in s.backends()}
            if backend not in known:
                raise ToolError(f"unknown backend {backend!r}; known: {', '.join(known)}")
            b = known[backend]
            if accept_licence is not None and accept_licence != b["licence_id"]:
                raise ToolError(f"accept_licence {accept_licence!r} does not match backend {backend!r}, whose licence is "
                                f"{b['licence_id']!r} ({b['licence']}, {b['licence_url']})")
            if not b["licence_accepted"] and accept_licence is None:
                raise ToolError(
                    f"LicenceNotAccepted: backend {backend!r} needs its weights licence {b['licence_id']!r} accepted: "
                    f"{b['licence']} ({b['licence_url']}), commercial use: {b['commercial_use']}. Ask the user to review "
                    f"it; only if they explicitly agree, call again with accept_licence={b['licence_id']!r}.")
            if not b["available"]:
                raise ToolError(f"backend {backend!r} is not available: {b['reason']}")
            ext = ".mid" if b.get("output") == "midi" else ".wav"
            return _new_out("ai", "gen", ext, out)

        dest = await _run(prep)
        params = {"backend": backend, "prompt": prompt, "seconds": seconds, "bpm": bpm, "key": key, "seed": seed,
                  "out": str(dest), "extra": {"model": model} if model else {}}
        cb = _progress_bridge(ctx)
        done = await _run(lambda: s.worker.run("generate", params, [accept_licence] if accept_licence else None,
                                               progress=cb, timeout=s.config.job_timeout))
        if accept_licence:
            s.backends_cache = None
        result = {k: done.get(k) for k in ("outputs", "backend", "model", "sample_rate", "seconds", "elapsed_s", "provenance")}
        if import_to_project:
            media = [o for o in done.get("outputs", []) if not o.endswith(".json")]
            result["import"] = await _run(_import_outputs, f"import generated {backend} audio", media, start_bar, track_id)
        return result

    @mcp.tool(annotations=AI)
    @guard
    async def separate_stems(
        input: Annotated[str, Field(description="Audio file to split")],
        out_dir: Annotated[str | None, Field(description="Directory for the stems (default: <project>/stems)")] = None,
        import_to_project: Annotated[bool, Field(description="Import every stem as its own audio track (one undo step)")] = False,
        start_bar: BarF = 1.0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Split an audio file into stems (drums, bass, vocals, other) with Demucs via the flux-ai worker.
        Optionally import the stems as new audio tracks."""
        src = s.path(input)
        if not src.is_file():
            raise ToolError(f"file not found: {src}")
        dest = s.path(out_dir, "out_dir") if out_dir else await _run(s.default_out_dir, "stems")
        done = await _run(lambda: s.worker.run("separate", {"input": str(src), "out": str(dest)},
                                               progress=_progress_bridge(ctx), timeout=s.config.job_timeout))
        result: dict = {"outputs": done.get("outputs", [])}
        if import_to_project and result["outputs"]:
            result["import"] = await _run(_import_outputs, f"import stems of {src.name}", result["outputs"], start_bar, None)
        return result

    @mcp.tool(annotations=AI)
    @guard
    async def transcribe_to_midi(
        input: Annotated[str, Field(description="Audio file to transcribe (works best on a single instrument or stem)")],
        out_dir: Annotated[str | None, Field(description="Directory for the .mid file (default: <project>/midi)")] = None,
        import_to_project: Annotated[bool, Field(description="Import the MIDI into the project (one undo step)")] = True,
        start_bar: BarF = 1.0,
        track_id: Annotated[str | None, Field(description="Instrument/MIDI track to import into (default: a new track)")] = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio to MIDI notes with Basic Pitch via the flux-ai worker, then optionally import it."""
        src = s.path(input)
        if not src.is_file():
            raise ToolError(f"file not found: {src}")
        dest = s.path(out_dir, "out_dir") if out_dir else await _run(s.default_out_dir, "midi")
        done = await _run(lambda: s.worker.run("transcribe", {"input": str(src), "out": str(dest)},
                                               progress=_progress_bridge(ctx), timeout=s.config.job_timeout))
        result: dict = {"outputs": done.get("outputs", [])}
        mids = [o for o in result["outputs"] if o.lower().endswith((".mid", ".midi"))]
        if import_to_project and mids:
            result["import"] = await _run(_import_outputs, f"import transcription of {src.name}", mids, start_bar, track_id)
        return result

    def _require_longform() -> None:
        if not s.longform_available():
            raise ToolError("long-form generation is unavailable: this flux-ai has no 'longform' subcommand "
                            "(update flux-ai-bridge)")

    def _load_plan(done: dict) -> tuple[str | None, dict | None]:
        plan = done.get("plan")
        path = done.get("plan_path")
        if isinstance(plan, str):
            path, plan = plan, None
        if path is None:
            path = next((o for o in done.get("outputs", []) if o.endswith(".json") and "provenance" not in o), None)
        if plan is None and path and Path(path).is_file():
            try:
                plan = json.loads(Path(path).read_text())
            except (OSError, json.JSONDecodeError):
                plan = None
        return path, plan

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False))
    @guard
    async def plan_longform(
        genre: Annotated[str, Field(min_length=1, max_length=100, description="e.g. 'deep house', 'ambient', 'drum and bass'")],
        hours: Annotated[float, Field(gt=0, le=24, description="Total length in hours")],
        seed: Annotated[int, Field(ge=0)] = 0,
        out_dir: Annotated[str | None, Field(description="Where to write the plan (default: <project>/longform)")] = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Plan a long-form piece (a set of sections/tracks with tempo, key and energy over time) without
        generating audio. The plan is written to plan.json and exposed as the flux://plan resource. Review
        it with the user, then call run_longform with its path."""
        await _run(_require_longform)
        dest = s.path(out_dir, "out_dir") if out_dir else await _run(s.default_out_dir, "longform")
        args = ["longform", "--genre", genre, "--hours", str(hours), "--plan-only", "--seed", str(seed), "--out", str(dest)]
        done = await _run(lambda: run_oneshot(args, s.config.flux_ai, _progress_bridge(ctx), 3600))
        path, plan = _load_plan(done)
        s.latest_plan, s.latest_plan_path = plan, path
        brief = plan
        if isinstance(plan, dict) and len(json.dumps(plan)) > 20000:
            brief = {k: (v if len(json.dumps(v)) < 2000 else f"<{type(v).__name__}, see flux://plan>") for k, v in plan.items()}
        return {"plan_path": path, "plan": brief, "outputs": done.get("outputs", [])}

    @mcp.tool(annotations=AI)
    @guard
    async def run_longform(
        plan_path: Annotated[str | None, Field(description="plan.json from plan_longform (recommended). If omitted, genre and hours are required")] = None,
        genre: Annotated[str | None, Field(max_length=100)] = None,
        hours: Annotated[float | None, Field(gt=0, le=24)] = None,
        neural_backend: Annotated[str | None, Field(description="Generation backend for audio parts (from list_backends); its licence must already be accepted")] = None,
        seed: Annotated[int, Field(ge=0)] = 0,
        out_dir: Annotated[str | None, Field(description="Output directory (default: <project>/longform); must not contain a previous run")] = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Render a long-form piece from a plan with the flux-ai long-form orchestrator. This can take a long
        time (minutes to hours) and reports progress. It writes many files under out_dir. Model licences are
        not accepted here: if the backend's licence is not yet accepted it fails; get the user's agreement
        and accept it via generate_audio first."""
        await _run(_require_longform)
        args = ["longform"]
        if plan_path:
            pp = s.path(plan_path, "plan_path")
            if not pp.is_file():
                raise ToolError(f"plan not found: {pp}")
            args += ["--plan", str(pp)]
        elif genre and hours:
            args += ["--genre", genre, "--hours", str(hours)]
        else:
            raise ToolError("give plan_path (from plan_longform), or both genre and hours")
        dest = s.path(out_dir, "out_dir") if out_dir else await _run(s.default_out_dir, "longform")
        args += ["--seed", str(seed), "--out", str(dest)]
        if neural_backend:
            args += ["--neural-backend", neural_backend]
        done = await _run(lambda: run_oneshot(args, s.config.flux_ai, _progress_bridge(ctx), s.config.job_timeout))
        path, plan = _load_plan(done)
        if plan is not None:
            s.latest_plan, s.latest_plan_path = plan, path
        return {k: v for k, v in done.items() if k not in ("event", "protocol_version", "job_id")}

    # --------------------------------------------------------------- resources
    @mcp.resource("flux://project", name="project", title="Open project",
                  description="describe_project JSON of the open project ({\"open\": false} if none)",
                  mime_type="application/json")
    async def project_resource() -> str:
        def go():
            try:
                proj = s.describe()
            except Exception as e:
                return {"open": False, "error": _error_text(e)}
            return {"open": False} if proj is None else {"open": True, **proj}
        return json.dumps(await _run(go), indent=1)

    @mcp.resource("flux://plan", name="plan", title="Long-form plan",
                  description="The latest long-form plan from plan_longform/run_longform in this session",
                  mime_type="application/json")
    async def plan_resource() -> str:
        if s.latest_plan is None and s.latest_plan_path is None:
            return json.dumps({"plan": None, "hint": "No plan yet: call plan_longform."})
        return json.dumps({"path": s.latest_plan_path, "plan": s.latest_plan}, indent=1)

    @mcp.resource("flux://backends", name="backends", title="AI backends",
                  description="flux-ai generation backends, availability and licence status",
                  mime_type="application/json")
    async def backends_resource() -> str:
        try:
            return json.dumps({"backends": await _run(s.backends)}, indent=1)
        except Exception as e:
            return json.dumps({"backends": [], "error": _error_text(e)})

    mcp.session = s  # type: ignore[attr-defined]  # for tests and the eval runners
    return mcp
