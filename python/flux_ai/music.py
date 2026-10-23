"""Small music-theory and WAV helpers shared by backends and the benchmark (numpy only)."""
from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {n: i for i, n in enumerate(PITCH_CLASSES)}
_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0})


def parse_key(key: str | None) -> tuple[int, str] | None:
    """'A minor', 'Am', 'F# maj', 'Bb min', 'C' -> (pitch class, 'major'|'minor')."""
    if not key:
        return None
    m = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]?)\s*(.*?)\s*", key)
    if not m:
        raise ValueError(f"cannot parse key '{key}'")
    name = m.group(1).upper() + m.group(2).replace("♯", "#").replace("♭", "b")
    rest = m.group(3).lower()
    if rest in ("", "maj", "major", "ionian", "M"):
        mode = "major"
    elif rest in ("m", "min", "minor", "aeolian", "-"):
        mode = "minor"
    else:
        raise ValueError(f"cannot parse key mode '{rest}' in '{key}'")
    return _PC[name], mode


def key_name(pc: int, mode: str) -> str:
    return f"{PITCH_CLASSES[pc % 12]} {mode}"


def normalise_key(key: str | None) -> str | None:
    p = parse_key(key)
    return key_name(*p) if p else None


def chord_progression(pc: int, mode: str) -> list[list[int]]:
    """Four triads (pitch classes) for a 4-bar loop: i-VI-III-VII in minor, I-V-vi-IV in major."""
    if mode == "minor":
        roots, quals = [0, 8, 3, 10], ["min", "maj", "maj", "maj"]
    else:
        roots, quals = [0, 7, 9, 5], ["maj", "maj", "min", "maj"]
    out = []
    for r, q in zip(roots, quals):
        third = 3 if q == "min" else 4
        out.append([(pc + r) % 12, (pc + r + third) % 12, (pc + r + 7) % 12])
    return out


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float audio (frames, channels) or (frames,) as PCM. 24-bit via soundfile if present, else 16-bit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    x = np.clip(x, -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(str(path), x, sample_rate, subtype="PCM_24")
        return
    except ImportError:
        pass
    pcm = (x * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read audio as float32 (frames, channels)."""
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x, sr
    except ImportError:
        pass
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {width} (install soundfile)")
    return x.reshape(-1, ch), sr
