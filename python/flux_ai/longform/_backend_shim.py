# SPDX-License-Identifier: GPL-3.0-or-later
"""TEMPORARY stand-in for `flux_ai.backends` (phase P4, built in parallel).

Same names and signatures as the agreed contract; the long-form code imports
`flux_ai.backends` first and only falls back to this file, so it can be deleted
once the real package is merged. Only the `sine-test` backend does anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class GenerateRequest:
    prompt: str
    seconds: float
    out: Path
    seed: int = 0
    bpm: float | None = None
    key: str | None = None
    mode: str = "generate"
    context_audio: Path | None = None
    lyrics: str | None = None
    instrumental: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class GenerateResult:
    audio: Path
    sample_rate: int
    seconds: float
    backend: str
    model: str
    provenance: dict


class Backend:
    name = "base"

    def available(self) -> tuple[bool, str]:
        return False, "not implemented"

    def generate(self, req: GenerateRequest, progress: Callable | None = None) -> GenerateResult:
        raise NotImplementedError


class SineTestBackend(Backend):
    """Deterministic test signal: a soft tonic drone plus a click on every beat."""

    name = "sine-test"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def generate(self, req: GenerateRequest, progress: Callable | None = None) -> GenerateResult:
        import soundfile as sf

        sr = 48000
        n = int(req.seconds * sr)
        t = np.arange(n) / sr
        tonic_hz = 110.0
        if req.key:
            names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
            alias = {"Db": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10}
            tn = req.key.split()[0]
            pc = names.index(tn) if tn in names else alias.get(tn, 9)
            tonic_hz = 440.0 * 2 ** ((pc - 9) / 12) / 2
        x = 0.2 * np.sin(2 * np.pi * tonic_hz * t)
        if req.bpm:
            beat = 60.0 / req.bpm
            ph = (t % beat)
            x += 0.5 * np.sin(2 * np.pi * 1000 * t) * np.exp(-ph / 0.01) * np.clip(ph / 0.001, 0, 1)
        x *= np.clip(t / 0.01, 0, 1) * np.clip((req.seconds - t) / 0.01, 0, 1)
        out = Path(req.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.stack([x, x], 1).astype(np.float32), sr, subtype="PCM_24")
        if progress:
            progress(1.0)
        return GenerateResult(out, sr, req.seconds, self.name, "sine-test-0",
                              {"backend": self.name, "model": "sine-test-0", "licence": "GPL-3.0-or-later",
                               "seed": req.seed, "prompt": req.prompt})


class _Unavailable(Backend):
    def __init__(self, name: str):
        self.name = name

    def available(self) -> tuple[bool, str]:
        return False, f"backend {self.name!r} not installed (flux_ai.backends missing; shim only has sine-test)"


def get_backend(name: str) -> Backend:
    if name == "sine-test":
        return SineTestBackend()
    return _Unavailable(name)
