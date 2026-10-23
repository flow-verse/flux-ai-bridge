# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact beat grid shared by the MIDI writer, the renderer and the timeline.

Tempo is stored per beat as an integer number of microseconds (what a MIDI
set_tempo event holds), so every component computes identical times and two
overlapping sections line up to the sample. Tempo changes between sections are
linear per-beat ramps across the transition overlap; both sides compute the
ramp with the same function, so their beats coincide.
"""
from __future__ import annotations

from dataclasses import dataclass

BEATS_PER_BAR = 4
TICKS_PER_BEAT = 960


def ramp_bpm(a: float, b: float, k: int, n: int) -> float:
    return a + (b - a) * (k + 0.5) / n


def us_per_beat(bpm: float) -> int:
    return int(round(60_000_000 / bpm))


def overlap_bars(plan: dict, i: int) -> int:
    """Bars by which section i overlaps section i+1 (0 for 'continue' or the last section)."""
    secs = plan["sections"]
    if i < 0 or i >= len(secs) - 1:
        return 0
    t = secs[i].get("transition_out") or {"type": "continue"}
    if t.get("type", "continue") == "continue":
        return 0
    return int(min(t.get("bars", 0), secs[i]["bars"], secs[i + 1]["bars"]))


def section_beat_us(plan: dict, i: int) -> list[int]:
    """Per-beat durations (µs) of section i, including tempo ramps in its overlaps."""
    secs = plan["sections"]
    s = secs[i]
    bpm = float(s["bpm"])
    beats = s["bars"] * BEATS_PER_BAR
    tempos = [bpm] * beats
    n_in = overlap_bars(plan, i - 1) * BEATS_PER_BAR if i > 0 else 0
    n_out = overlap_bars(plan, i) * BEATS_PER_BAR
    if n_in:
        prev = float(secs[i - 1]["bpm"])
        if prev != bpm:
            for k in range(n_in):
                tempos[k] = ramp_bpm(prev, bpm, k, n_in)
    if n_out:
        nxt = float(secs[i + 1]["bpm"])
        if nxt != bpm:
            for k in range(n_out):
                tempos[beats - n_out + k] = ramp_bpm(bpm, nxt, k, n_out)
    return [us_per_beat(t) for t in tempos]


def beat_starts_us(beat_us: list[int]) -> list[int]:
    out, acc = [], 0
    for d in beat_us:
        out.append(acc)
        acc += d
    out.append(acc)
    return out


def tick_to_us(tick: int, starts: list[int], beat_us: list[int]) -> float:
    b, frac = divmod(tick, TICKS_PER_BEAT)
    if b >= len(beat_us):  # past the end: continue at the last tempo
        return starts[-1] + (tick - len(beat_us) * TICKS_PER_BEAT) * beat_us[-1] / TICKS_PER_BEAT
    return starts[b] + frac * beat_us[b] / TICKS_PER_BEAT


@dataclass
class Placement:
    index: int
    start_us: int          # global start of bar 0
    dur_us: int            # body length (bars only, no tail)
    overlap_in_us: int     # overlap with the previous section
    overlap_out_us: int    # overlap with the next section
    overlap_in_bars: int
    overlap_out_bars: int
    bar_starts_us: list[int]  # relative to section start, len bars+1


def timeline(plan: dict) -> list[Placement]:
    out: list[Placement] = []
    start = 0
    for i, s in enumerate(plan["sections"]):
        beat_us = section_beat_us(plan, i)
        starts = beat_starts_us(beat_us)
        bar_starts = starts[:: BEATS_PER_BAR]
        n_in = overlap_bars(plan, i - 1) if i > 0 else 0
        n_out = overlap_bars(plan, i)
        dur = starts[-1]
        ov_in = bar_starts[n_in] if n_in else 0
        ov_out = dur - bar_starts[s["bars"] - n_out] if n_out else 0
        if i > 0:
            prev = out[-1]
            start = prev.start_us + prev.dur_us - prev.overlap_out_us
            # The overlap must be the same duration seen from both sides.
            assert prev.overlap_out_us == ov_in, (i, prev.overlap_out_us, ov_in)
        out.append(Placement(i, start, dur, ov_in, ov_out, n_in, n_out, bar_starts))
    return out


def total_us(plan: dict) -> int:
    t = timeline(plan)
    return t[-1].start_us + t[-1].dur_us if t else 0


def us_to_samples(us: float, sr: int) -> int:
    return int(round(us * sr / 1_000_000))
