# SPDX-License-Identifier: GPL-3.0-or-later
"""Section renderers: section MIDI + section plan -> audio file.

`BuiltinSynthRenderer` is a pure numpy/scipy placeholder so the pipeline runs
anywhere, faster than real time. `FluxHostRenderer` (fluxhost.py) renders the
pitched layers through flux-host's headless engine (`flux-engine serve`).

Contract: `render(section_midi, section_plan, out_wav)` writes the stereo mix
of the section to `out_wav` (format from the extension; FLAC/WAV) starting at
bar 0 of the section, `body + tail_seconds` long, and may also write a
`<stem>.low.<ext>` file holding only kick + bass (used for bass-swap
transitions; without it the transition engine splits the mix with a
crossover). Samples are stored scaled by `RenderResult.scale` (to fit 24-bit
storage); divide by it to get the real level.
"""
from __future__ import annotations

import abc
import bisect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import genres, synth, timing

MELODIC = ("bass", "chords", "pad", "lead")
DRUM_BY_NOTE = {36: "kick", 37: "perc", 39: "clap", 42: "hat_closed", 46: "hat_open", 51: "ride", 49: "crash"}
DRUM_PAN = {"kick": 0.0, "clap": 0.0, "hat_closed": 0.25, "hat_open": -0.2, "perc": -0.35, "ride": 0.3, "crash": -0.1}


@dataclass
class RenderResult:
    mix: Path
    low: Path | None
    sample_rate: int
    frames: int
    body_frames: int
    scale: float = 1.0
    stats: dict = field(default_factory=dict)


@dataclass
class ParsedNote:
    track: str
    channel: int
    start: float   # seconds from section start
    end: float
    pitch: float
    vel: float
    bright: float  # 0..1 from CC74 at note-on


@dataclass
class SynthOutput:
    low: np.ndarray      # (n, 2) float32: kick + bass
    high: np.ndarray     # (n, 2) float32: everything else, effects included
    duck: np.ndarray     # (n,) sidechain gain from the kick (1 = no ducking)
    frames: int
    body_frames: int
    tail: float
    notes: int


class Renderer(abc.ABC):
    name = "abstract"
    version = "0"

    def __init__(self, sample_rate: int = 48000, tail_seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.tail_seconds = tail_seconds

    def identity(self) -> dict:
        return {"name": self.name, "version": self.version, "sample_rate": self.sample_rate, "tail_seconds": self.tail_seconds}

    @abc.abstractmethod
    def render(self, section_midi: Path, section_plan: dict, out_wav: Path) -> RenderResult: ...

    def close(self) -> None:
        """Release external resources (engine processes). Safe to call more than once."""


def parse_midi(path: Path) -> tuple[list[ParsedNote], float]:
    """Notes with absolute times (tempo map honoured) and the body length in seconds."""
    import mido

    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    # tempo map from every track (we write it in track 0)
    tempos: list[tuple[int, int]] = []
    end_tick = 0
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append((tick, msg.tempo))
        end_tick = max(end_tick, tick) if tr.name == "tempo" else end_tick
    tempos.sort()
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))
    seg_ticks = [t for t, _ in tempos]
    seg_us = [0.0]
    for (t0, us0), (t1, _) in zip(tempos, tempos[1:]):
        seg_us.append(seg_us[-1] + (t1 - t0) * us0 / tpb)

    def to_s(tick: int) -> float:
        i = bisect.bisect_right(seg_ticks, tick) - 1
        return (seg_us[i] + (tick - seg_ticks[i]) * tempos[i][1] / tpb) / 1e6

    notes: list[ParsedNote] = []
    for tr in mid.tracks:
        name = tr.name
        if name == "tempo":
            continue
        tick = 0
        bend = [0.0] * 16
        rng_ = [2.0] * 16
        bright = [0.5] * 16
        rpn = [(127, 127)] * 16
        active: dict[tuple[int, int], list] = {}
        for msg in tr:
            tick += msg.time
            if msg.type == "pitchwheel":
                bend[msg.channel] = msg.pitch / 8192 * rng_[msg.channel]
            elif msg.type == "control_change":
                ch = msg.channel
                if msg.control == 101:
                    rpn[ch] = (msg.value, rpn[ch][1])
                elif msg.control == 100:
                    rpn[ch] = (rpn[ch][0], msg.value)
                elif msg.control == 6 and rpn[ch] == (0, 0):
                    rng_[ch] = float(msg.value)
                elif msg.control == 74:
                    bright[ch] = msg.value / 127
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append(
                    [tick, msg.note + bend[msg.channel], msg.velocity / 127, bright[msg.channel]])
            elif msg.type in ("note_off", "note_on"):
                lst = active.get((msg.channel, msg.note))
                if lst:
                    st, pitch, vel, br = lst.pop(0)
                    notes.append(ParsedNote(name, msg.channel, to_s(st), to_s(tick), pitch, vel, br))
    notes.sort(key=lambda n: (n.start, n.track))
    return notes, to_s(end_tick)


class BuiltinSynthRenderer(Renderer):
    """Pure numpy/scipy renderer (placeholder for flux-host + Surge XT)."""

    name = "builtin-synth"
    version = "1"

    def identity(self) -> dict:
        # include a digest of the DSP code so cached renders are invalidated when it changes
        import hashlib

        src = Path(synth.__file__).read_bytes() + Path(__file__).read_bytes()
        return {**super().identity(), "code": hashlib.sha256(src).hexdigest()[:12]}

    def tail_for(self, section_plan: dict) -> float:
        """Seconds rendered after the body: the reverb tail plus a margin, at least tail_seconds."""
        tim = genres.load(section_plan["genre"])["timbre"]
        reverb_s = float(tim.get("reverb", {}).get("seconds", 1.8))
        return max(self.tail_seconds, min(reverb_s + 1.0, 8.0))

    def synthesize(self, notes: list[ParsedNote], body_s: float, section_plan: dict,
                   tracks: set[str] | None = None) -> SynthOutput:
        """Render `notes` (all tracks, or only those in `tracks`) to unscaled float32 buffers.

        Also returns the kick-driven sidechain envelope, so a caller that renders
        some layers elsewhere (FluxHostRenderer) can duck them the same way.
        """
        sr = self.sample_rate
        g = genres.load(section_plan["genre"])
        tim = g["timbre"]
        rng = np.random.default_rng(section_plan["seeds"].get("render", section_plan["seeds"]["section"]))
        if tracks is not None:
            notes = [x for x in notes if x.track in tracks]
        reverb_s = float(tim.get("reverb", {}).get("seconds", 1.8))
        tail = self.tail_for(section_plan)
        body = int(round(body_s * sr))
        n = body + int(tail * sr)
        low = np.zeros((n, 2), dtype=np.float32)    # kick + bass
        high = np.zeros((n, 2), dtype=np.float32)   # everything else (dry)
        rev_send = np.zeros((n, 2), dtype=np.float32)
        dly_send = np.zeros((n, 2), dtype=np.float32)
        music = np.zeros((n, 2), dtype=np.float32)  # ducked bus (chords/pad/lead)
        bass = np.zeros((n, 2), dtype=np.float32)

        # drums: a few pre-rendered variants per instrument
        shots: dict[str, list[np.ndarray]] = {}
        kick_on, kick_vel = [], []
        for nt in (x for x in notes if x.track == "drums"):
            inst = DRUM_BY_NOTE.get(int(round(nt.pitch)))
            if inst is None:
                continue
            if inst not in shots:
                shots[inst] = [synth.drum_oneshot(inst, tim.get(inst, {}), sr, rng) for _ in range(4)]
            x = shots[inst][int(rng.integers(0, 4))]
            s0 = int(round(nt.start * sr))
            v = nt.vel ** 1.6
            gl, gr = synth.pan_gains(DRUM_PAN.get(inst, 0.0))
            if inst == "kick":
                synth.add_at(low, x * v, s0, gl, gr)
                kick_on.append(s0)
                kick_vel.append(nt.vel)
            else:
                synth.add_at(high, x * v, s0, gl, gr)
                if inst in ("clap", "perc"):
                    synth.add_at(rev_send, x * v * 0.25, s0, gl, gr)

        # melodic voices
        for nt in (x for x in notes if x.track in MELODIC):
            p = tim.get(nt.track, {})
            y = synth.voice(p, synth.midi_to_hz(nt.pitch), nt.end - nt.start, nt.vel, nt.bright, sr, rng)
            y *= synth.db(p.get("level_db", -18))
            s0 = int(round(nt.start * sr))
            pan = 0.0 if nt.track == "bass" else float(rng.uniform(-0.45, 0.45)) if nt.track in ("chords", "pad") else float(rng.uniform(-0.15, 0.15))
            gl, gr = synth.pan_gains(pan)
            if nt.track == "bass":
                synth.add_at(bass, y, s0, gl, gr)
            else:
                synth.add_at(music, y, s0, gl, gr)
                if p.get("reverb"):
                    synth.add_at(rev_send, y * float(p["reverb"]), s0, gl, gr)
                if p.get("delay"):
                    synth.add_at(dly_send, y * float(p["delay"]), s0, gl, gr)

        # sidechain pump on bass and music from the kick
        sc = tim.get("sidechain", {})
        duck = synth.duck_envelope(np.array(kick_on, dtype=np.int64), np.array(kick_vel), n, sr,
                                   float(sc.get("release", 0.16)), float(sc.get("depth", 0.0)))
        bass *= duck[:, None]
        music *= duck[:, None]
        low += bass
        high += music

        # delay (dotted eighth at the section tempo) and reverb
        bpm = float(section_plan["bpm"])
        if np.any(dly_send):
            high += synth.pingpong(dly_send, int(round(0.75 * 60 / bpm * sr)), sr=sr)
        if np.any(rev_send):
            ir = synth.reverb_ir(reverb_s, sr, rng)
            high += synth.convolve_stereo(rev_send, ir) * synth.db(-6)
        tex = tim.get("texture", "none")
        if tex and tex != "none":
            high += synth.texture(tex, n, sr, rng)
        if tim.get("lowpass_master"):
            for c in range(2):
                high[:, c] = synth.lp(high[:, c], float(tim["lowpass_master"]), sr).astype(np.float32)

        return SynthOutput(low, high, duck, n, body, tail, len(notes))

    def render(self, section_midi: Path, section_plan: dict, out_wav: Path) -> RenderResult:
        notes, body_s = parse_midi(Path(section_midi))
        o = self.synthesize(notes, body_s, section_plan)
        return write_section(out_wav, o.low, o.high, self.sample_rate, o.body_frames, {"notes": o.notes, "tail_seconds": o.tail})


def write_section(out_wav: Path, low: np.ndarray, high: np.ndarray, sr: int, body: int, stats: dict) -> RenderResult:
    """Fade the tail end, scale to fit 24-bit storage, write `out_wav` and `<stem>.low.<ext>`."""
    import soundfile as sf

    # fade the very end of the tail so cached audio always ends at zero
    f = min(int(0.5 * sr), len(low))
    ramp = np.linspace(1, 0, f, dtype=np.float32)[:, None]
    low[-f:] *= ramp
    high[-f:] *= ramp
    mix = low + high
    peak = float(max(np.max(np.abs(mix)), np.max(np.abs(low)), 1e-9))
    scale = min(1.0, 0.98 / peak)
    out_wav = Path(out_wav)
    low_path = out_wav.with_name(out_wav.stem + ".low" + out_wav.suffix)
    sub = "PCM_24"
    sf.write(str(out_wav), mix * scale, sr, subtype=sub)
    sf.write(str(low_path), low * scale, sr, subtype=sub)
    return RenderResult(out_wav, low_path, sr, len(mix), body, scale, {**stats, "peak": peak})


RENDERERS = {"builtin": BuiltinSynthRenderer, "builtin-synth": BuiltinSynthRenderer}


def get_renderer(name: str, **kw) -> Renderer:
    if name == "flux-host":
        from .fluxhost import FluxHostRenderer

        return FluxHostRenderer(**kw)
    try:
        return RENDERERS[name](**kw)
    except KeyError:
        raise KeyError(f"unknown renderer {name!r}; choose from {sorted([*RENDERERS, 'flux-host'])}") from None
