# SPDX-License-Identifier: GPL-3.0-or-later
"""Transition engine and timeline assembler.

Sections are placed on one global grid (see timing.py). Where a transition
overlaps two sections, both were rendered with the same per-beat tempo ramp,
so their bars coincide to the sample. Each section arrives as a mix plus a
`low` stem (kick + bass); the high band is mix - low, so the two bands always
sum back to the original exactly.

- blend: high band equal-power crossfade over the overlap; with bass_swap the
  low band swaps over one beat at a bar of the overlap (`swap_at`, a fraction
  of the overlap, default 0.5 = the middle bar; 0 = the first bar, so the new
  key's bass and kick arrive with the first incoming chords and the outgoing
  bass never plays under them). Outgoing low out, incoming low in, never both
  at once. With eq=false both bands follow the high-band curve.
- the set ends with an END_FADE_BARS fade to silence over the last bars of the
  final section, so a plan that stops on a bar line does not cut off.
- fade: equal-power crossfade of the full mix over the overlap.
- continue: no overlap; the next section starts at the bar line and the
  previous section's release tail plays out under it.

`Timeline.read(a, b)` returns any range of the assembled (pre-master) mix, so
the master stage can stream over hours without holding them in memory.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import timing


@dataclass
class SectionAudio:
    index: int
    mix: Path
    low: Path
    scale: float
    frames: int


@dataclass
class _Placed:
    index: int
    audio: SectionAudio
    start: int          # global sample of bar 0
    body: int           # body length in samples
    ov_in: int          # incoming overlap length in samples (0 = none)
    ov_out: int
    in_type: str
    out_type: str
    in_swap: bool
    out_swap: bool
    in_eq: bool
    out_eq: bool
    swap_at: int | None   # global sample of incoming bass swap (relative to this section's start: start + ...)
    swap_out_at: int | None
    swap_w_in: int
    swap_w_out: int


END_FADE_BARS = 4


def _s(us: float, sr: int) -> int:
    return timing.us_to_samples(us, sr)


class Timeline:
    def __init__(self, plan: dict, audio: list[SectionAudio], sr: int, max_open: int = 8):
        self.plan = plan
        self.sr = sr
        self.max_open = max_open
        self._files: OrderedDict = OrderedDict()
        places = timing.timeline(plan)
        secs = plan["sections"]
        self.placed: list[_Placed] = []
        swap_info: dict[int, tuple[int, int]] = {}
        for p in places:
            i = p.index
            if i > 0 and p.overlap_in_bars:
                prev_t = secs[i - 1]["transition_out"]
                frac = float(prev_t.get("swap_at", 0.5))
                k = min(p.overlap_in_bars - 1, max(0, int(round(p.overlap_in_bars * frac))))
                if frac >= 0.5 and p.overlap_in_bars > 1:
                    k = max(1, k)
                beat_us = timing.section_beat_us(plan, i)
                w_us = beat_us[min(len(beat_us) - 1, k * timing.BEATS_PER_BAR)]
                at = _s(p.start_us + p.bar_starts_us[k], sr)
                swap_info[i] = (at, max(1, _s(w_us, sr)))
        for p in places:
            i = p.index
            s = secs[i]
            prev_t = secs[i - 1]["transition_out"] if i > 0 else {"type": "continue"}
            t = s["transition_out"] if i < len(secs) - 1 else {"type": "continue"}
            in_type = prev_t.get("type", "continue") if p.overlap_in_bars else "continue"
            out_type = t.get("type", "continue") if p.overlap_out_bars else "continue"
            si = swap_info.get(i)
            so = swap_info.get(i + 1)
            self.placed.append(_Placed(
                i, audio[i], _s(p.start_us, sr), _s(p.dur_us, sr),
                _s(p.overlap_in_us, sr) if in_type != "continue" else 0,
                _s(p.overlap_out_us, sr) if out_type != "continue" else 0,
                in_type, out_type,
                bool(prev_t.get("bass_swap", False)) and in_type == "blend",
                bool(t.get("bass_swap", False)) and out_type == "blend",
                bool(prev_t.get("eq", True)), bool(t.get("eq", True)),
                si[0] if si else None, so[0] if so else None,
                si[1] if si else 1, so[1] if so else 1))
        # The overlap regions must match sample-for-sample from both sides.
        for a, b in zip(self.placed, self.placed[1:]):
            if a.ov_out:
                b_end = a.start + a.body
                if b.start + b.ov_in != b_end or b.start != b_end - a.ov_out:
                    # rounding of independent us->sample conversions; pin to the incoming side
                    a.ov_out = b_end - b.start
                    b.ov_in = a.ov_out
        last = self.placed[-1]
        self.total = max(p.start + p.audio.frames for p in self.placed)
        self.end_of_body = last.start + last.body
        lp = places[-1]
        nb = len(lp.bar_starts_us) - 1
        self.fade_from = last.start + _s(lp.bar_starts_us[max(0, nb - END_FADE_BARS)], sr)

    # ------------------------------------------------------------------ gains
    def gains(self, p: _Placed, rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(low_gain, high_gain) for relative sample indices of section p."""
        gl = np.ones(len(rel))
        gh = np.ones(len(rel))
        if p.ov_in:
            m = rel < p.ov_in
            u = rel[m] / p.ov_in
            if p.in_type == "fade" or not p.in_eq:
                gh[m] = np.sin(np.pi / 2 * u)
                gl[m] = gh[m]
            else:
                gh[m] = np.sin(np.pi / 2 * u)
                if p.in_swap and p.swap_at is not None:
                    gl[m] = np.clip((rel[m] + p.start - p.swap_at) / p.swap_w_in, 0, 1)
                else:
                    gl[m] = gh[m]
        if p.ov_out:
            o0 = p.body - p.ov_out
            m = (rel >= o0) & (rel < p.body)
            u = (rel[m] - o0) / p.ov_out
            ho = np.cos(np.pi / 2 * u)
            if p.out_type == "fade" or not p.out_eq:
                lo = ho
            elif p.out_swap and p.swap_out_at is not None:
                lo = 1 - np.clip((rel[m] + p.start - p.swap_out_at) / p.swap_w_out, 0, 1)
            else:
                lo = ho
            gh[m] *= ho
            gl[m] *= lo
            after = rel >= p.body
            gh[after] = 0.0
            gl[after] = 0.0
        if p.index == 0:  # 5 ms safety fade at the very start
            f = int(0.005 * self.sr)
            m = rel < f
            gh[m] *= rel[m] / f
            gl[m] *= rel[m] / f
        if p.index == self.placed[-1].index:  # end of set: fade the last bars, silence after the body
            g0 = rel + p.start
            m = g0 >= self.fade_from
            span = max(1, self.end_of_body - self.fade_from)
            w = np.cos(np.pi / 2 * np.clip((g0[m] - self.fade_from) / span, 0, 1))
            gh[m] *= w
            gl[m] *= w
        return gl, gh

    # ------------------------------------------------------------------ I/O
    def _open(self, path: Path):
        import soundfile as sf

        key = str(path)
        f = self._files.get(key)
        if f is None:
            f = sf.SoundFile(key)
            self._files[key] = f
            while len(self._files) > self.max_open:
                _, old = self._files.popitem(last=False)
                old.close()
        else:
            self._files.move_to_end(key)
        return f

    def _read(self, path: Path, r0: int, r1: int) -> np.ndarray:
        f = self._open(path)
        f.seek(r0)
        x = f.read(r1 - r0, dtype="float32", always_2d=True)
        if len(x) < r1 - r0:
            x = np.concatenate([x, np.zeros((r1 - r0 - len(x), x.shape[1]), np.float32)])
        return x

    def read(self, a: int, b: int) -> np.ndarray:
        """Assembled mix for global samples [a, b) (zeros outside the set)."""
        out = np.zeros((b - a, 2), dtype=np.float64)
        for p in self.placed:
            s0, s1 = p.start, p.start + p.audio.frames
            if s1 <= a or s0 >= b:
                continue
            r0, r1 = max(a, s0) - s0, min(b, s1) - s0
            rel = np.arange(r0, r1)
            gl, gh = self.gains(p, rel)
            if not np.any(gl) and not np.any(gh):
                continue
            mix = self._read(p.audio.mix, r0, r1).astype(np.float64)
            low = self._read(p.audio.low, r0, r1).astype(np.float64)
            y = (gl[:, None] * low + gh[:, None] * (mix - low)) / p.audio.scale
            o0 = r0 + s0 - a
            out[o0:o0 + (r1 - r0)] += y
        return out

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    # ------------------------------------------------------------------ grid
    def joins(self) -> list[dict]:
        """Boundary sample positions worth checking for discontinuities."""
        out = []
        for a, b in zip(self.placed, self.placed[1:]):
            out.append({"from": a.index, "to": b.index, "type": a.out_type, "start": b.start,
                        "end_of_prev_body": a.start + a.body, "end_of_prev_file": a.start + a.audio.frames,
                        "swap": b.swap_at})
        return out

    def bar_starts(self) -> list[int]:
        """Global sample of every bar, overlap bars counted once (outgoing side)."""
        out = []
        for p, pl in zip(timing.timeline(self.plan), self.placed):
            first = p.overlap_in_bars if pl.ov_in else 0
            for k in range(first, len(p.bar_starts_us) - 1):
                out.append(_s(p.start_us + p.bar_starts_us[k], self.sr))
        out.append(self.end_of_body)
        return out
