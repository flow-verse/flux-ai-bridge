# SPDX-License-Identifier: GPL-3.0-or-later
"""FluxHostRenderer: render each section's pitched layers through flux-host's headless engine.

Per section, over one warm ``flux-engine serve`` process (JSON-RPC 2.0, see
flux-host/flux/engine/API.md):

1. ``create_project`` in ``<section dir>/project`` at the section tempo;
2. one agent turn adding an instrument track per layer (bass, chords, pad, lead,
   and drums if a drum instrument is configured), each with its plugin;
3. one agent turn writing each layer's notes (``write_notes``, converted from
   the section MIDI) and setting the range to the section body plus tail;
4. ``get_history``, ``save_project`` and ``render`` (32-bit float WAV with
   per-track stems) of exactly ``body + tail`` seconds from bar 1.

Timing. The project runs at the section's constant tempo. Note positions are
taken in seconds from the section MIDI (whose tempo map carries the per-beat
ramps across transitions) and converted to beats at that constant tempo, so the
rendered audio lines up with the plan's microsecond grid exactly as the built-in
synth's does. Tempo-synced plugin features (LFO or delay sync) follow the
constant tempo and drift slightly inside a ramp.

Drums. Unless a drum instrument is configured (``--flux-plugin drums=ID`` or
``flux_host.plugins.drums`` in the genre table), drums keep the built-in synth
(with its drum reverb and the genre's texture) and are mixed with the engine
stems. The built-in kick also drives the genre's sidechain ducking of the engine
layers, as it does in the built-in renderer. With a drum instrument the kick
(GM note 36) gets its own track so the ``low`` stem (kick + bass) for bass-swap
transitions can still be formed.

What the engine API cannot carry yet (rounded or dropped, counted in the section
stats): pitch bend (maqam quarter tones are rounded to the nearest semitone),
CC74 brightness automation, and presets (``load_preset`` does not exist, so a
plugin plays its default patch).

Output contract: identical to BuiltinSynthRenderer (`render.py`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .. import engine_client
from . import genres, render, synth

LAYERS = ("drums", "bass", "chords", "pad", "lead")
PITCHED = ("bass", "chords", "pad", "lead")
DEFAULT_PLUGIN = "Surge XT"
PLUGIN_ENV = "FLUX_LONGFORM_PLUGIN"
TICKS = 960  # write_notes positions are rounded to the engine's PPQ
KICK_NOTE = 36


def parse_plugin_specs(specs: list[str] | None) -> dict[str, str]:
    """`["bass=Surge XT", "default=Test Synth"]` -> {"bass": "Surge XT", "default": "Test Synth"}."""
    out: dict[str, str] = {}
    for spec in specs or []:
        layer, sep, pid = spec.partition("=")
        layer, pid = layer.strip().lower(), pid.strip()
        if not sep or not pid:
            raise ValueError(f"--flux-plugin wants LAYER=PLUGIN_ID, got {spec!r}")
        if layer not in (*LAYERS, "default"):
            raise ValueError(f"--flux-plugin layer {layer!r} is not one of {', '.join((*LAYERS, 'default'))}")
        out[layer] = pid
    return out


def notes_to_clips(notes: list[render.ParsedNote], bpm: float, clip_beats: float, tracks: dict[str, str]
                   ) -> tuple[dict[str, list[dict]], dict]:
    """Convert parsed section notes (seconds) into `write_notes` note lists per engine track.

    `tracks` maps a MIDI track name (layer) to the engine track it goes to; layers
    not in it are skipped. Drums may be split: a `"kick"` entry receives GM note 36
    and `"drums"` the rest. Times become beats at the constant `bpm`, rounded to
    1/960 beat; every note keeps at least one tick and ends inside the clip.
    """
    beat_per_s = bpm / 60.0
    clip_ticks = int(math.floor(clip_beats * TICKS + 1e-6))
    out: dict[str, list[dict]] = {name: [] for name in set(tracks.values())}
    stats = {"notes": 0, "microtones_rounded": 0, "clamped": 0}
    for nt in notes:
        layer = nt.track
        if layer == "drums" and "kick" in tracks and int(round(nt.pitch)) == KICK_NOTE:
            layer = "kick"
        dest = tracks.get(layer)
        if dest is None:
            continue
        pitch = int(round(nt.pitch))
        if abs(nt.pitch - pitch) > 0.01:
            stats["microtones_rounded"] += 1
        a = int(round(nt.start * beat_per_s * TICKS))
        b = int(round(nt.end * beat_per_s * TICKS))
        if a >= clip_ticks:
            stats["clamped"] += 1
            continue
        b = min(max(b, a + 1), clip_ticks)
        out[dest].append({"pitch": min(max(pitch, 0), 127), "start_beat": a / TICKS, "length_beats": (b - a) / TICKS,
                          "velocity": min(max(int(round(nt.vel * 127)), 1), 127)})
        stats["notes"] += 1
    return out, stats


class FluxHostRenderer(render.Renderer):
    """Renders pitched layers with CLAP/VST3 instruments in flux-host's headless engine."""

    name = "flux-host"
    version = "1"

    def __init__(self, sample_rate: int = 48000, tail_seconds: float = 4.0, engine: str | None = None,
                 plugins: dict[str, str] | None = None, block: int | None = None,
                 client: engine_client.EngineClient | None = None, call_timeout: float = 120.0,
                 render_timeout: float = 3600.0):
        super().__init__(sample_rate=sample_rate, tail_seconds=tail_seconds)
        self.plugins = dict(plugins or {})
        self.default_plugin = self.plugins.pop("default", None)
        self.engine = engine
        self.block = block
        self.call_timeout = call_timeout
        self.render_timeout = render_timeout
        self._client = client
        self._catalog: list[dict] | None = None
        self._identity: dict | None = None
        self.drum_synth = render.BuiltinSynthRenderer(sample_rate=sample_rate, tail_seconds=tail_seconds)
        self.restarts = 0

    # ------------------------------------------------------------------ engine
    @property
    def client(self) -> engine_client.EngineClient:
        if self._client is None:
            args = ["--sr", str(self.sample_rate)] + (["--block", str(self.block)] if self.block else [])
            self._client = engine_client.EngineClient(self.engine, args)
        return self._client

    def _call(self, method: str, params: dict | None = None, timeout: float | None = None):
        return self.client.call(method, params or {}, timeout or self.call_timeout)

    def ready_info(self) -> dict:
        c = self.client
        with c._lock:
            c.ensure_started()
            return dict(c._ready or {})

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def available(self) -> tuple[bool, str]:
        try:
            self.identity()
        except Exception as e:
            return False, str(e)
        return True, ""

    # ----------------------------------------------------------------- plugins
    def catalog(self) -> list[dict]:
        if self._catalog is None:
            self._catalog = self._call("list_plugins", {"instruments_only": True}, 600.0).get("plugins", [])
        return self._catalog

    def resolve(self, spec: str) -> dict:
        """Find an instrument by exact id or (case-insensitive) name."""
        cat = self.catalog()
        hit = next((p for p in cat if p.get("id") == spec), None) or \
            next((p for p in cat if str(p.get("name", "")).lower() == spec.lower()), None)
        if hit is None:
            names = ", ".join(sorted({p.get("name", "?") for p in cat})) or "none"
            raise RuntimeError(f"instrument {spec!r} not found by flux-engine (instruments found: {names}). Install it, "
                               f"add its folder to FLUX_CLAP_PATH, or choose another with --flux-plugin LAYER=ID")
        return {k: hit.get(k) for k in ("id", "name", "vendor", "version", "format")}

    def default_spec(self, genre: dict | None = None) -> str:
        gmap = ((genre or {}).get("flux_host") or {}).get("plugins") or {}
        return self.default_plugin or gmap.get("default") or os.environ.get(PLUGIN_ENV) or DEFAULT_PLUGIN

    def layer_specs(self, genre: dict) -> dict[str, str | None]:
        """Plugin spec per layer. Precedence: --flux-plugin LAYER, genre flux_host.plugins LAYER,
        --flux-plugin default, genre default, $FLUX_LONGFORM_PLUGIN, Surge XT. Drums never fall back to the
        default instrument: without an explicit drum instrument they use the built-in synth (None)."""
        gmap = (genre.get("flux_host") or {}).get("plugins") or {}
        out: dict[str, str | None] = {}
        for layer in LAYERS:
            spec = self.plugins.get(layer) or gmap.get(layer)
            if spec is None and layer != "drums":
                spec = self.default_spec(genre)
            out[layer] = spec
        return out

    def identity(self) -> dict:
        """Everything that changes the audio: engine build, plugins, this code. Starts the engine."""
        if self._identity is None:
            exe = Path(self.client.path)
            st = exe.stat()
            ready = self.ready_info()
            if ready.get("sample_rate") not in (None, self.sample_rate):
                raise RuntimeError(f"flux-engine runs at {ready.get('sample_rate')} Hz, the plan needs {self.sample_rate}")
            plugins = {layer: self.resolve(spec) for layer, spec in self.plugins.items()}
            plugins["default"] = self.resolve(self.default_spec())
            src = b"".join(Path(m.__file__).read_bytes() for m in (render, synth)) + Path(__file__).read_bytes()
            self._identity = {
                **super().identity(),
                "code": hashlib.sha256(src).hexdigest()[:12],
                "engine": {"api_version": ready.get("api_version"), "binary": str(exe), "size": st.st_size,
                           "mtime": int(st.st_mtime), "block": self.block},
                "plugins": plugins,
                "drums": "plugin" if "drums" in self.plugins else "builtin-synth unless the genre table maps drums",
            }
        return self._identity

    # ------------------------------------------------------------------ render
    def render(self, section_midi: Path, section_plan: dict, out_wav: Path) -> render.RenderResult:
        sr = self.sample_rate
        out_wav = Path(out_wav)
        genre = genres.load(section_plan["genre"])
        specs = self.layer_specs(genre)
        resolved = {layer: (self.resolve(spec) if spec else None) for layer, spec in specs.items()}
        drums_builtin = resolved["drums"] is None
        notes, body_s = render.parse_midi(Path(section_midi))
        # the built-in synth renders the drums (or nothing) and provides the kick sidechain envelope
        base = self.drum_synth.synthesize(notes, body_s, section_plan, tracks={"drums"})
        n, body = base.frames, base.body_frames
        if not drums_builtin:
            base.low[:] = 0
            base.high[:] = 0
        bpm = float(section_plan["bpm"])
        clip_beats = n / sr * bpm / 60.0
        clip_bars = int(math.ceil(clip_beats / 4 - 1e-9))
        routing = {layer: layer for layer in PITCHED}
        if not drums_builtin:
            routing.update({"drums": "drums", "kick": "kick"})
        clips, conv = notes_to_clips(notes, bpm, clip_bars * 4, routing)
        clips = {name: ns for name, ns in clips.items() if ns}
        plugin_of = {name: resolved["drums" if name == "kick" else name] for name in clips}

        last: Exception | None = None
        for attempt in range(2):
            try:
                eng = self._render_engine(section_plan["id"], out_wav.parent, bpm, clip_bars, n, clips, plugin_of)
                break
            except engine_client.EngineCrashed as e:  # restarted on the next call; redo the section from scratch
                last = e
                self.restarts += 1
        else:
            raise RuntimeError(f"flux-engine crashed twice rendering {section_plan['id']}: {last}") from last

        stems, info = eng
        level = {k: synth.db(v) for k, v in self.layer_gains(genre).items()}
        low, high = base.low, base.high
        duck = base.duck[:, None]
        for name, x in stems.items():
            if name in ("kick", "drums"):
                (low if name == "kick" else high)[:] += x * level.get("drums", 1.0)
            else:
                y = x * (level.get(name, 1.0) * duck)
                (low if name == "bass" else high)[:] += y
        stats = {"notes": len(notes), "tail_seconds": base.tail,
                 "plugins": {layer: (r or "builtin-synth") for layer, r in resolved.items()
                             if layer in clips or (layer == "drums" and drums_builtin)},
                 "engine": info, "conversion": {**conv, "pitch_bend": "not carried (rounded to semitones)",
                                                "cc74_brightness": "not carried"}}
        return render.write_section(out_wav, low, high, sr, body, stats)

    def layer_gains(self, genre: dict) -> dict[str, float]:
        """dB applied to each engine stem: genre flux_host.gain_db, else the genre timbre's level_db relative
        to the bass (keeps the table's balance between layers; the plugin's own output sets the level)."""
        explicit = (genre.get("flux_host") or {}).get("gain_db") or {}
        tim = genre.get("timbre", {})
        ref = float(tim.get("bass", {}).get("level_db", 0.0))
        out = {}
        for layer in LAYERS:
            if layer in explicit:
                out[layer] = float(explicit[layer])
            elif layer == "drums":
                out[layer] = 0.0
            else:
                out[layer] = float(tim.get(layer, {}).get("level_db", ref)) - ref
        return out

    def _render_engine(self, sid: str, workdir: Path, bpm: float, clip_bars: int, frames: int,
                       clips: dict[str, list[dict]], plugin_of: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict]:
        import soundfile as sf

        sr = self.sample_rate
        proj = workdir / "project"
        rdir = workdir / "engine-render"
        shutil.rmtree(proj, ignore_errors=True)
        shutil.rmtree(rdir, ignore_errors=True)
        rdir.mkdir(parents=True)
        t0 = time.time()
        self._call("create_project", {"dir": str(proj), "bpm": bpm})
        turns = []
        track_ids: dict[str, str] = {}

        def turn(label: str, body) -> None:
            self._call("begin_agent_turn", {"label": label})
            try:
                body()
            except engine_client.EngineError:
                try:
                    self._call("end_agent_turn", {"discard": True})
                except Exception:
                    pass
                raise
            res = self._call("end_agent_turn")
            turns.append({"label": f"Agent: {res.get('label', label)}", "recorded": res.get("recorded")})

        def add_tracks() -> None:
            for name in clips:
                res = self._call("add_track", {"type": "instrument", "name": name, "plugin_id": plugin_of[name]["id"]})
                st = (res.get("plugin") or {}).get("status")
                if st != "ok":
                    raise RuntimeError(f"{sid}: plugin {plugin_of[name]['name']} on track {name} has status {st}")
                track_ids[name] = res["track"]

        def write() -> None:
            for name, notes in clips.items():
                self._call("write_notes", {"track": track_ids[name], "start_bar": 1, "length_bars": clip_bars,
                                           "notes": notes, "name": f"{sid} {name}"}, 300.0)
            self._call("set_range", {"start_bar": 1, "end_bar": 1 + clip_bars})

        turn(f"longform {sid}: instruments", add_tracks)
        turn(f"longform {sid}: notes", write)
        history = self._call("get_history")
        self._call("save_project")
        build_s = time.time() - t0
        res = self._call("render", {"out": str(rdir / "mix.wav"), "start_sec": 0.0, "end_sec": frames / sr,
                                    "bits": "32f", "stems": True}, self.render_timeout)
        if int(res.get("sample_rate", sr)) != sr:
            raise RuntimeError(f"{sid}: engine rendered at {res.get('sample_rate')} Hz, expected {sr}")
        by_id = {v: k for k, v in track_ids.items()}
        stems: dict[str, np.ndarray] = {}
        for o in res.get("outputs", []):
            name = by_id.get(o.get("track"))
            if name is None:
                continue
            x = sf.read(o["path"], dtype="float32", always_2d=True)[0]
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            x = x[:frames, :2]
            if len(x) < frames:
                x = np.pad(x, ((0, frames - len(x)), (0, 0)))
            stems[name] = x
        missing = set(track_ids) - set(stems)
        if missing:
            raise RuntimeError(f"{sid}: flux-engine returned no stem for {sorted(missing)}")
        shutil.rmtree(rdir, ignore_errors=True)
        info = {"api_version": (self.client._ready or {}).get("api_version"), "project": "project",
                "turns": turns, "history": history, "tracks": {k: plugin_of[k]["id"] for k in track_ids},
                "build_seconds": round(build_s, 3), "render_wall_seconds": round(float(res.get("wall_sec", 0)), 3),
                "render_speed_factor": res.get("speed_factor"), "engine_restarts": self.client.restarts,
                "stem_rms_dbfs": {k: round(float(10 * np.log10(np.mean(np.square(v, dtype=np.float64)) + 1e-20)), 2)
                                  for k, v in stems.items()}}
        return stems, json.loads(json.dumps(info, default=str))
