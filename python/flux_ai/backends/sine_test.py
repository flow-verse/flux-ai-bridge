"""Deterministic built-in backend for tests and CI: no weights, numpy only.

It renders a 4-bar chord loop in the requested key (i-VI-III-VII in minor,
I-V-vi-IV in major) with a kick on every beat and a bass note on beats 1 and 3,
at the requested BPM. The seed picks the arpeggio voicing, so outputs are
reproducible per seed. ``continue`` appends a continuation after the context
audio on the same beat grid; ``repaint``/``infill`` regenerate
``extra["start"]..extra["end"]`` seconds of the context.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from flux_ai import music, provenance
from flux_ai.backends import GenerateRequest, GenerateResult, check_request, finalize, report

SR = 44100


def _midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def render(seconds: float, bpm: float, key: str, seed: int, t0: float = 0.0, sr: int = SR) -> np.ndarray:
    """Stereo float32 (frames, 2) starting at absolute time t0 on the beat grid."""
    pc, mode = music.parse_key(key)
    chords = music.chord_progression(pc, mode)
    rng = np.random.default_rng(seed)
    voicing = rng.permutation(3)
    phases = rng.uniform(0, 2 * np.pi, 3)
    n = int(round(seconds * sr))
    t = t0 + np.arange(n) / sr
    beat = 60.0 / bpm
    beat_idx = np.floor(t / beat).astype(np.int64)
    in_beat = t - beat_idx * beat
    bar = beat_idx // 4
    out = np.zeros(n, dtype=np.float64)
    # pad: sustained triad, one chord per bar
    for c in range(4):
        sel = (bar % 4) == c
        if not sel.any():
            continue
        for k, p in enumerate(chords[c]):
            f = _midi_hz(60 + p)
            out[sel] += 0.08 * np.sin(2 * np.pi * f * t[sel] + phases[k])
        # arpeggio (eighth notes), seed-dependent order
        eighth = np.floor(in_beat[sel] / (beat / 2)).astype(int)
        notes = np.asarray(chords[c])[voicing[(beat_idx[sel] * 2 + eighth) % 3]]
        env = np.exp(-6.0 * (in_beat[sel] % (beat / 2)))
        out[sel] += 0.10 * env * np.sin(2 * np.pi * _midi_hz(72) * 2 ** (notes / 12) * t[sel])
        # bass on beats 1 and 3
        bass_on = (beat_idx[sel] % 2) == 0
        fb = _midi_hz(36 + chords[c][0])
        benv = np.exp(-3.0 * in_beat[sel]) * bass_on
        out[sel] += 0.25 * benv * np.sin(2 * np.pi * fb * t[sel])
    # kick on every beat: pitch-swept sine burst
    kenv = np.exp(-in_beat / 0.06)
    kphase = 2 * np.pi * (50 * in_beat + 60 * 0.03 * (1 - np.exp(-in_beat / 0.03)))
    out += 0.6 * kenv * np.sin(kphase)
    out = np.tanh(out)
    return np.stack([out, out], axis=1).astype(np.float32)


class SineTestBackend:
    name = "sine-test"
    licence_id = "none"
    modes = frozenset({"generate", "continue", "repaint", "infill"})
    models = ["sine-test-1"]
    output_kind = "audio"

    def available(self) -> tuple[bool, str]:
        return True, "built-in"

    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult:
        check_request(self, req)
        started = provenance.now()
        bpm = float(req.bpm or 120.0)
        key = req.key or "C major"
        report(progress, 0.0, "rendering")
        if req.mode == "generate":
            audio = render(req.seconds, bpm, key, req.seed)
        else:
            ctx, sr = music.read_wav(req.context_audio)
            if sr != SR:
                raise ValueError(f"sine-test context must be {SR} Hz, got {sr}")
            if ctx.shape[1] == 1:
                ctx = np.repeat(ctx, 2, axis=1)
            ctx_s = ctx.shape[0] / SR
            if req.mode == "continue":
                audio = np.concatenate([ctx[:, :2], render(req.seconds, bpm, key, req.seed, t0=ctx_s)])
            else:
                start = float(req.extra.get("start", 0.0))
                end = float(req.extra.get("end", min(ctx_s, start + req.seconds)))
                a, b = int(start * SR), int(end * SR)
                audio = ctx[:, :2].copy()
                audio[a:b] = render((b - a) / SR, bpm, key, req.seed, t0=start)[: b - a]
        report(progress, 0.8, "writing wav")
        music.write_wav(req.out, audio, SR)
        report(progress, 1.0, "done")
        return finalize(
            self, req, output=req.out, model="sine-test-1", model_version="1", sample_rate=SR,
            seconds=audio.shape[0] / SR, started_at=started, backend_params={"bpm": bpm, "key": key},
        )
