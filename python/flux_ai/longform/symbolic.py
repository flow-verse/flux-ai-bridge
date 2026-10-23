# SPDX-License-Identifier: GPL-3.0-or-later
"""Symbolic generator: one section of the plan -> a standard MIDI file.

Drums, bass, chords, pad and a lead built from the set-level motif library.
Everything is deterministic for a given plan section (seeds come from the
plan). Microtonal pitches (maqam quarter tones) are written as per-note pitch
bends with a +-2 semitone bend range, rotating notes over a pool of channels
so polyphonic parts can carry different bends (MPE-style).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from . import genres, theory, timing

GENERATOR_VERSION = "1"
TPB = timing.TICKS_PER_BEAT
STEP = TPB // 4  # one sixteenth
BAR = TPB * timing.BEATS_PER_BAR

GM_DRUMS = {"kick": 36, "perc": 37, "clap": 39, "hat_closed": 42, "hat_open": 46, "ride": 51, "crash": 49}
DRUM_LAYERS = ["kick", "clap", "hat_closed", "hat_open", "perc", "ride"]
CHANNELS = {"drums": [9], "bass": [0], "lead": [1, 2], "chords": [3, 4, 5, 6, 7, 8], "pad": [10, 11, 12, 13, 14, 15]}
BEND_RANGE = 2.0  # semitones
CC_BRIGHTNESS = 74


@dataclass
class Note:
    track: str
    start: int      # ticks from section start
    dur: int        # ticks
    pitch: float    # float MIDI pitch (drums: GM note)
    vel: float      # 0..1


def _vel_char(ch: str) -> float | None:
    return {"x": 0.85, "X": 1.0, "o": 0.5}.get(ch)


def _swing_tick(step: int, swing: float) -> int:
    t = step * STEP
    if step % 2 == 1:
        t += int(round((swing - 0.5) * TPB / 2))
    return t


def _transform(motif: dict, name: str) -> list[list]:
    notes = [list(n) for n in motif["notes"]]
    steps = 16 * motif["bars"]
    if name == "up2":
        for n in notes:
            n[2] += 2
    elif name == "down3":
        for n in notes:
            n[2] -= 3
    elif name == "invert":
        first = notes[0][2]
        for n in notes:
            n[2] = 2 * first - n[2]
    elif name == "retrograde":
        degs = [n[2] for n in notes][::-1]
        for n, d in zip(notes, degs):
            n[2] = d
    elif name == "shift":
        for n in notes:
            n[0] = (n[0] + 2) % steps
        notes.sort(key=lambda n: n[0])
    return notes


def _vary_motif(notes: list[list], steps: int, variation: int, rng: random.Random) -> list[list]:
    """Per-phrase variation so repeated phrases are never identical copies."""
    out = [list(n) for n in notes]
    kind = variation % 4
    if kind == 1 and len(out) > 2:  # change the ending
        out[-1][2] += rng.choice([-2, -1, 1, 2])
    elif kind == 2 and len(out) > 3:  # drop a note, lengthen its neighbour
        k = rng.randrange(1, len(out))
        dropped = out.pop(k)
        out[k - 1][1] = min(steps - out[k - 1][0], out[k - 1][1] + dropped[1])
    elif kind == 3:  # answer phrase: up a third
        for n in out:
            n[2] += 2
    if rng.random() < 0.3 and out:  # ornament: neighbour-note grace
        k = rng.randrange(len(out))
        if out[k][1] >= 2:
            out[k][1] -= 1
            out.insert(k + 1, [out[k][0] + out[k][1], 1, out[k][2] + rng.choice([-1, 1]), out[k][3] * 0.8])
    return out


def _fold(p: float, lo: float, hi: float) -> float:
    while p > hi:
        p -= 12
    while p < lo:
        p += 12
    return p


def generate_notes(plan: dict, index: int) -> tuple[list[Note], dict]:
    s = plan["sections"][index]
    g = genres.load(s.get("genre", plan["genre"]))
    key = s["key"]
    cents = theory.scale_cents(key, g.get("maqamat"))
    swing = float(g.get("swing", 0.5))
    layers = set(s.get("layers", []))
    bars_ = s["bars"]
    order = [x for x in ("kick", "hat_closed", "bass", "perc", "hat_open", "clap", "ride", "chords", "pad", "lead") if x in layers]
    if len(order) > 2:  # entry/exit order differs per section so transitions do not all sound alike
        rest = order[1:]
        random.Random(s["seeds"]["section"] * 3 + 1).shuffle(rest)
        order = order[:1] + rest
    n_ph = (bars_ + 7) // 8

    def active(layer: str, b: int) -> bool:
        """DJ-style intros/outros: layers enter (intro) or leave (outro) across 8-bar phrases."""
        if layer not in layers:
            return False
        if n_ph < 2 or s["type"] not in ("intro", "outro") or len(order) < 2:
            return True
        # never fewer than two layers: a bare kick is identical in every segment
        n_on = 2 + round((b // 8) * (len(order) - 2) / (n_ph - 1))
        if s["type"] == "outro":
            n_on = len(order) + 2 - n_on
        return order.index(layer) < n_on

    seeds = s["seeds"]
    bars = s["bars"]
    bp = s["bar_plan"]
    notes: list[Note] = []
    cc: list[tuple[int, str, int]] = []  # (tick, track, value) brightness automation

    # ---- drums
    pat = g["drums"]["patterns"][s["patterns"]["drums"]]
    fill = g["drums"].get("fill", {})
    for b in range(bars):
        e = bp[b]["energy"]
        rng = random.Random(seeds.get("drums", 0) * 7919 + bp[b]["variation"] * 31 + b)
        # phrase-level arrangement mutation (8-bar phrases), so long grooves keep moving
        # consecutive phrases (and phrases two apart) always get different mutations
        mut = (b // 8 + seeds["section"]) % 6
        for inst in DRUM_LAYERS:
            if not active(inst, b):
                continue
            line = pat.get(inst, "") or ""
            if bp[b]["fill"] and fill.get(inst):
                line = fill[inst]
            elif line:
                if mut == 1 and inst == "hat_open":
                    line = ""
                elif mut == 2 and inst == "perc":
                    line = line[-2:] + line[:-2]
                elif mut == 3 and inst == "hat_closed":
                    line = "".join("o" if c == "." and k % 2 == 1 else c for k, c in enumerate(line))
                elif mut == 4 and inst == "clap" and b % 4 == 3:
                    line = line[:12] + "...."
                elif mut == 5 and inst == "perc":
                    line = "".join("o" if k in (3, 11) else c for k, c in enumerate(line))
            if not line:
                continue
            chars = list(line)
            if inst in ("hat_closed", "perc"):
                for k in range(16):
                    if chars[k] == "." and rng.random() < 0.12 * e:
                        chars[k] = "o"
                    elif chars[k] != "." and k % 4 != 0 and rng.random() < 0.06:
                        chars[k] = "."
            for k, ch in enumerate(chars):
                v = _vel_char(ch)
                if v is None:
                    continue
                jitter = 0 if inst == "kick" else rng.randint(-6, 6)
                vel = min(1.0, v * (0.78 + 0.22 * e) + rng.uniform(-0.05, 0.05))
                start = b * BAR + max(0, _swing_tick(k, swing) + jitter)
                notes.append(Note("drums", start, STEP // 2, GM_DRUMS[inst], vel))
        if s["type"] == "peak" and b % 16 == 0 and e > 0.7 and "ride" in g["drums"]["patterns"][s["patterns"]["drums"]] and "hat_open" in layers:
            notes.append(Note("drums", b * BAR, STEP, GM_DRUMS["crash"], 0.6))

    # ---- bass
    if "bass" in layers:
        bline = g["bass"]["patterns"][s["patterns"]["bass"]]
        base = theory.tonic_midi(key, int(g["bass"].get("octave", 1)))
        while base < 28:
            base += 12
        offsets = {"R": 0, "3": 2, "5": 4, "7": 6, "8": 7, "b": -1}
        for b in range(bars):
            if not active("bass", b):
                continue
            rng = random.Random(seeds.get("bass", 0) * 31 + bp[b]["variation"] + b)
            root = bp[b]["chord"]
            bmut = (b // 8 + seeds["section"] // 6) % 5
            line = bline
            if bmut == 1 and bline[0] != "R":  # shift an offbeat line by an eighth
                line = bline[-2:] + bline[:-2]
            elif bmut == 2:  # octave answers on the last beat
                line = bline[:12] + "".join("8" if c == "R" else c for c in bline[12:])
            elif bmut == 3 and bline[14] == ".":  # pickup fifth into the next bar
                line = bline[:14] + "5" + bline[15]
            elif bmut == 4:  # leave the last beat of every other bar empty
                line = bline if b % 2 == 0 else bline[:12] + "...."
            k = 0
            while k < 16:
                ch = line[k]
                if ch in offsets:
                    ties = 0
                    while k + 1 + ties < 16 and line[k + 1 + ties] == "-":
                        ties += 1
                    off = offsets[ch]
                    if ch == "R" and rng.random() < 0.08 * bp[b]["energy"]:
                        off = 7
                    pitch = theory.degree_pitch(base, cents, root + off)
                    pitch = _fold(pitch, base - 5, base + 12)
                    dur = int((1 + ties) * STEP * 0.85)
                    vel = min(1.0, 0.75 + 0.2 * bp[b]["energy"] + rng.uniform(-0.04, 0.04))
                    notes.append(Note("bass", b * BAR + _swing_tick(k, swing), dur, pitch, vel))
                    k += 1 + ties
                else:
                    k += 1

    # ---- chords
    tones = list(g["chords"].get("tones", [0, 2, 4]))
    ext = g["chords"].get("extension")
    c_base = theory.tonic_midi(key, int(g["chords"].get("octave", 4)))
    if "chords" in layers:
        rnames = sorted(g["chords"]["rhythms"])
        main_r = s["patterns"]["chord_rhythm"]
        alt_r = rnames[(rnames.index(main_r) + 1) % len(rnames)] if main_r in rnames else main_r
        for b in range(bars):
            if not active("chords", b):
                continue
            rng = random.Random(seeds.get("chords", 0) * 17 + bp[b]["variation"] + b)
            e = bp[b]["energy"]
            ph = b // 8
            # every third phrase answers with the neighbouring rhythm; voicing register cycles
            # with period 5, so with the drum (6) and bass (5) mutations a phrase pattern only
            # recurs after 30 phrases (240 bars), longer than any section
            rline = g["chords"]["rhythms"][alt_r if (ph + seeds["section"]) % 3 == 2 else main_r]
            tt = tones + ([ext] if ext is not None and e > 0.6 and bp[b]["variation"] % 3 != 0 else [])
            lo = c_base - 5 + (0, 3, -2, 5, 1)[ph % 5]
            pitches = sorted({_fold(theory.degree_pitch(c_base, cents, bp[b]["chord"] + t), lo, lo + 14) for t in tt})
            k = 0
            while k < 16:
                if rline[k] == "x":
                    ties = 0
                    while k + 1 + ties < 16 and rline[k + 1 + ties] == "-":
                        ties += 1
                    if not (ties == 0 and rng.random() < 0.1):  # occasionally skip a stab
                        dur = int((1 + ties) * STEP * (0.95 if ties else 0.6))
                        vel = min(1.0, 0.6 + 0.3 * e + rng.uniform(-0.05, 0.05))
                        for p in pitches:
                            notes.append(Note("chords", b * BAR + _swing_tick(k, swing), dur, p, vel))
                    k += 1 + ties
                else:
                    k += 1

    # ---- pad: held chord tones, one chord per run of equal bar chords
    if "pad" in layers:
        b = 0
        while b < bars:
            run = 1
            while b + run < bars and bp[b + run]["chord"] == bp[b]["chord"] and run < 4:
                run += 1
            pitches = sorted({_fold(theory.degree_pitch(c_base, cents, bp[b]["chord"] + t), c_base - 3, c_base + 12) for t in (0, 2, 4)})
            for p in pitches:
                notes.append(Note("pad", b * BAR, run * BAR - STEP, p, 0.7))
            b += run

    # ---- lead from the motif library
    motif_ref = s.get("motif")
    if "lead" in layers and motif_ref:
        motif = next(m for m in plan["motifs"] if m["id"] == motif_ref["id"])
        base_notes = _transform(motif, motif_ref.get("transform", "identity"))
        mb = motif["bars"]
        l_base = theory.tonic_midi(key, int(g["lead"].get("octave", 5)))
        for b0 in range(0, bars, mb):
            ph = bp[b0]["phrase"]
            e = bp[b0]["energy"]
            # call/response: at lower energy the lead rests every other motif slot
            if e < 0.75 and (b0 // mb) % 2 == 1 and b0 % 8 != 0:
                continue
            rng = random.Random(seeds.get("lead", 0) * 13 + ph * 101 + b0)
            var = bp[b0]["variation"] if (b0 // mb) % 2 else bp[ph * 8 if ph * 8 < bars else b0]["variation"]
            phrase_notes = _vary_motif(base_notes, 16 * mb, var, rng)
            for st, ln, deg, vel in phrase_notes:
                tick = b0 * BAR + _swing_tick(st, swing)
                if tick >= bars * BAR:
                    continue
                pitch = theory.degree_pitch(l_base, cents, deg)
                dur = min(int(ln * STEP * 0.9), bars * BAR - tick)
                notes.append(Note("lead", tick, max(STEP // 2, dur), pitch, min(1.0, vel * (0.75 + 0.25 * e))))

    # ---- brightness automation: energy plus slow drift (never the same twice)
    drift = random.Random(seeds["section"])
    period, phase = drift.uniform(6, 14), drift.uniform(0, 2 * math.pi)
    for b in range(bars):
        val = 0.2 + 0.65 * bp[b]["energy"] + 0.12 * math.sin(2 * math.pi * b / period + phase)
        cc.append((b * BAR, "all", int(round(min(1.0, max(0.0, val)) * 127))))

    notes.sort(key=lambda n: (n.start, n.track, n.pitch))
    return notes, {"cc": cc, "bars": bars}


def write_midi(plan: dict, index: int, out_path: Path) -> dict:
    import mido

    notes, extra = generate_notes(plan, index)
    s = plan["sections"][index]
    beat_us = timing.section_beat_us(plan, index)
    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)

    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("track_name", name="tempo", time=0))
    tempo.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo.append(mido.MetaMessage("marker", text=f"{s['id']} {s['type']} {theory.key_str(s['key'])}", time=0))
    last_tick, last_us = 0, None
    for b, us in enumerate(beat_us):
        if us != last_us:
            tick = b * TPB
            tempo.append(mido.MetaMessage("set_tempo", tempo=us, time=tick - last_tick))
            last_tick, last_us = tick, us
    tempo.append(mido.MetaMessage("end_of_track", time=len(beat_us) * TPB - last_tick))
    mid.tracks.append(tempo)

    for track in ("drums", "bass", "chords", "pad", "lead"):
        tnotes = [n for n in notes if n.track == track]
        pool = CHANNELS[track]
        busy = {ch: -1 for ch in pool}
        bend = {ch: 0 for ch in pool}
        ev: list[tuple[int, int, object]] = []
        for ch in pool:
            if track != "drums":
                for c, v in ((101, 0), (100, 0), (6, int(BEND_RANGE)), (38, 0)):
                    ev.append((0, 0, mido.Message("control_change", channel=ch, control=c, value=v)))
        if track != "drums":
            for tick, _, val in extra["cc"]:
                for ch in pool:
                    ev.append((tick, 1, mido.Message("control_change", channel=ch, control=CC_BRIGHTNESS, value=val)))
        for n in tnotes:
            if track == "drums":
                ch, num, bv = 9, int(n.pitch), None
            else:
                num = int(round(n.pitch))
                frac = n.pitch - num
                bv = int(round(frac / BEND_RANGE * 8192))
                bv = max(-8192, min(8191, bv))
                free = [c for c in pool if busy[c] <= n.start and bend[c] == bv] or [c for c in pool if busy[c] <= n.start]
                ch = free[0] if free else min(pool, key=lambda c: busy[c])
                busy[ch] = n.start + n.dur
                if bend[ch] != bv:
                    ev.append((n.start, 2, mido.Message("pitchwheel", channel=ch, pitch=bv)))
                    bend[ch] = bv
            vel = max(1, min(127, int(round(n.vel * 127))))
            ev.append((n.start, 3, mido.Message("note_on", channel=ch, note=num, velocity=vel)))
            ev.append((n.start + max(1, n.dur), 0, mido.Message("note_off", channel=ch, note=num, velocity=0)))
        ev.sort(key=lambda e: (e[0], e[1]))
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage("track_name", name=track, time=0))
        prev = 0
        for tick, _, msg in ev:
            mt.append(msg.copy(time=tick - prev))
            prev = tick
        mid.tracks.append(mt)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))
    return {"path": str(out_path), "notes": len(notes)}
