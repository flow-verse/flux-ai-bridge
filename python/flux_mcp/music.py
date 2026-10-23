# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit conversions done by tools, not by the model (the DAWZY lesson).

- note names ("C4", "F#2", "Bb3") to MIDI pitch, with middle C = C4 = 60;
- dB to linear gain for gain-type plugin parameters;
- a compact project summary returned after every edit.
"""
from __future__ import annotations

import math
import re

_NOTE = re.compile(r"^\s*([A-Ga-g])([#♯b♭]*)(-?\d+)\s*$")
_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

SYSTEM_TRACK_TYPES = {"chord", "modulator", "marker"}


def pitch_to_midi(pitch: int | str) -> int:
    """MIDI number from an int (0-127) or a note name; C4 = 60 (scientific pitch notation)."""
    if isinstance(pitch, bool):
        raise ValueError("pitch must be a MIDI number or a note name like 'C4'")
    if isinstance(pitch, (int, float)):
        if float(pitch) != int(pitch):
            raise ValueError(f"pitch {pitch} is not a whole MIDI note number")
        n = int(pitch)
    else:
        s = str(pitch).strip()
        if re.fullmatch(r"-?\d+", s):
            n = int(s)
        else:
            m = _NOTE.match(s)
            if not m:
                raise ValueError(f"cannot read pitch '{pitch}': use a MIDI number 0-127 or a name like C4, F#2, Bb3 (C4 = 60)")
            letter, acc, octave = m.groups()
            n = _BASE[letter.upper()] + 12 * (int(octave) + 1)
            for a in acc:
                n += 1 if a in "#♯" else -1
    if not 0 <= n <= 127:
        raise ValueError(f"pitch '{pitch}' is MIDI {n}, outside 0-127")
    return n


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float | None:
    return 20.0 * math.log10(gain) if gain > 0 else None


def summarize(project: dict | None, history: dict | None = None) -> dict:
    """Compact project state for tool results: ids, names, plugins and clip extents (no notes)."""
    if not project:
        return {"open": False}
    tracks = []
    for t in project.get("tracks", []):
        if t.get("type") in SYSTEM_TRACK_TYPES:
            continue
        entry = {"id": t["id"], "name": t["name"], "type": t["type"]}
        if t.get("plugins"):
            entry["plugins"] = [{"id": p["id"], "name": p["name"], "slot": p.get("slot"), "status": p.get("status")}
                                for p in t["plugins"]]
        if t.get("clips"):
            entry["clips"] = [{k: c.get(k) for k in ("id", "name", "kind", "start_bar", "end_bar", "note_count") if k in c}
                              for c in t["clips"]]
        tracks.append(entry)
    out = {
        "open": True,
        "title": project.get("title"),
        "dir": project.get("dir"),
        "tempo": project.get("tempo"),
        "time_signature": {k: v for k, v in (project.get("time_signature") or {}).items() if k != "changes" or v},
        "range": project.get("range"),
        "markers": project.get("markers", []),
        "tracks": tracks,
    }
    if history is not None:
        out["history"] = {k: history.get(k) for k in ("can_undo", "can_redo", "undo_text", "redo_text")}
    return out
