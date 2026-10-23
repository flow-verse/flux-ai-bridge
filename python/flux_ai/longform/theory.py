# SPDX-License-Identifier: GPL-3.0-or-later
"""Pitch, scale and key helpers shared by the planner and the symbolic generator.

Every scale is a list of cents above the tonic (seven degrees), so Western modes
and maqamat with quarter tones go through the same code path. A pitch is a float
MIDI number (62.5 = a quarter tone above D4).
"""
from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_ALIASES = {"Db": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10, "Cb": 11, "E#": 5, "Fb": 4, "B#": 0}

WESTERN_MODES: dict[str, list[float]] = {
    "major": [0, 200, 400, 500, 700, 900, 1100],
    "minor": [0, 200, 300, 500, 700, 800, 1000],
    "dorian": [0, 200, 300, 500, 700, 900, 1000],
    "phrygian": [0, 100, 300, 500, 700, 800, 1000],
    "mixolydian": [0, 200, 400, 500, 700, 900, 1000],
    "lydian": [0, 200, 400, 600, 700, 900, 1100],
}
# Modes that sit on the minor side of the Camelot wheel.
MINOR_LIKE = {"minor", "dorian", "phrygian"}


def pitch_class(name: str) -> int:
    name = name.strip()
    if name in NOTE_NAMES:
        return NOTE_NAMES.index(name)
    if name in _ALIASES:
        return _ALIASES[name]
    raise ValueError(f"unknown note name {name!r}")


def note_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def key_str(key: dict) -> str:
    """{"tonic": "A", "mode": "minor"} -> "A minor" (the form generation backends take)."""
    return f"{key['tonic']} {key['mode']}"


def camelot(key: dict) -> str | None:
    """Camelot code for Western keys (8A = A minor, 8B = C major); None for maqamat."""
    mode = key["mode"]
    if mode not in WESTERN_MODES:
        return None
    pc = pitch_class(key["tonic"])
    if mode in MINOR_LIKE:
        return f"{(7 * pc + 4) % 12 + 1}A"
    return f"{(7 * pc + 7) % 12 + 1}B"


def from_camelot(code: str, minor_mode: str = "minor", major_mode: str = "major") -> dict:
    num, letter = int(code[:-1]), code[-1].upper()
    for pc in range(12):
        if letter == "A" and (7 * pc + 4) % 12 + 1 == num:
            return {"tonic": note_name(pc), "mode": minor_mode}
        if letter == "B" and (7 * pc + 7) % 12 + 1 == num:
            return {"tonic": note_name(pc), "mode": major_mode}
    raise ValueError(code)


def camelot_neighbours(code: str) -> list[tuple[str, str]]:
    """Harmonic-mixing compatible moves: (move name, target code)."""
    num, letter = int(code[:-1]), code[-1].upper()
    other = "B" if letter == "A" else "A"

    def wrap(n: int) -> int:
        return (n - 1) % 12 + 1

    return [
        ("same", code),
        ("+1", f"{wrap(num + 1)}{letter}"),
        ("-1", f"{wrap(num - 1)}{letter}"),
        ("relative", f"{num}{other}"),
        ("+2 energy boost", f"{wrap(num + 2)}{letter}"),
    ]


def scale_cents(key: dict, maqamat: dict | None = None) -> list[float]:
    mode = key["mode"]
    if maqamat and mode in maqamat:
        return list(maqamat[mode]["cents"])
    if mode in WESTERN_MODES:
        return WESTERN_MODES[mode]
    raise ValueError(f"unknown mode {mode!r}")


def degree_pitch(tonic_midi: float, cents: list[float], degree: int) -> float:
    """Scale degree (0-based, may be negative or > 6) to a float MIDI pitch."""
    octave, step = divmod(degree, len(cents))
    return tonic_midi + 12 * octave + cents[step] / 100.0


def tonic_midi(key: dict, octave: int) -> int:
    """MIDI note of the tonic in a given octave (C4 = 60)."""
    return 12 * (octave + 1) + pitch_class(key["tonic"])
