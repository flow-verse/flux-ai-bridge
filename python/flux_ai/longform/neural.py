# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional neural audio layer per section, via the P4 backend interface.

The layer covers the section's constant-tempo body (not its tempo-ramped
overlaps). Generated audio is tempo-checked, phase-aligned to the beat grid,
faded at bar boundaries and mixed under the symbolic layers. Any failure or an
unavailable backend skips the layer and is recorded in provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import theory, timing


def backend_api():
    """(GenerateRequest, get_backend, source). Real package first, shim second."""
    try:
        from flux_ai.backends import GenerateRequest, get_backend  # type: ignore
        return GenerateRequest, get_backend, "flux_ai.backends"
    except ImportError:
        from ._backend_shim import GenerateRequest, get_backend
        return GenerateRequest, get_backend, "flux_ai.longform._backend_shim"


def layer_region(plan: dict, index: int) -> tuple[int, int, float]:
    """(start_bar, end_bar, seconds) of the constant-tempo body of a section."""
    s = plan["sections"][index]
    b0 = timing.overlap_bars(plan, index - 1) if index > 0 else 0
    b1 = s["bars"] - timing.overlap_bars(plan, index)
    if b1 - b0 < 2:
        b0, b1 = 0, s["bars"]
    starts = timing.beat_starts_us(timing.section_beat_us(plan, index))[:: timing.BEATS_PER_BAR]
    return b0, b1, (starts[b1] - starts[b0]) / 1e6


def _onset_env(x: np.ndarray, sr: int, hop: int = 512):
    from scipy import signal

    _, _, z = signal.stft(x, sr, nperseg=2048, noverlap=2048 - hop, boundary=None, padded=False)
    mag = np.log1p(np.abs(z))
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
    flux = flux - flux.mean()
    return np.maximum(flux, 0), sr / hop


def estimate_tempo_phase(x: np.ndarray, sr: int, bpm: float) -> dict:
    env, fps = _onset_env(x, sr)
    if len(env) < fps * 4 or not np.any(env):
        return {"bpm_est": None, "tempo_ok": False, "offset_s": 0.0}
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 55)
    k = lo + int(np.argmax(ac[lo:hi]))
    bpm_est = 60 * fps / k
    ok = any(abs(bpm_est * m - bpm) / bpm < 0.04 for m in (1, 2, 0.5))
    period = fps * 60 / bpm
    scores = []
    for off in range(int(period)):
        idx = (off + np.arange(0, len(env) / period - 1) * period).astype(int)
        scores.append(env[idx[idx < len(env)]].sum())
    offset_frames = int(np.argmax(scores))
    return {"bpm_est": round(float(bpm_est), 2), "tempo_ok": bool(ok), "offset_s": offset_frames / fps}


_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(x: np.ndarray, sr: int) -> dict:
    from scipy import signal

    f, _, z = signal.stft(x[: sr * 60], sr, nperseg=8192)
    mag = np.abs(z).mean(axis=1)
    chroma = np.zeros(12)
    m = (f > 50) & (f < 4000)
    pcs = (np.round(12 * np.log2(f[m] / 440.0)) + 9) % 12
    np.add.at(chroma, pcs.astype(int), mag[m])
    best = None
    for pc in range(12):
        for name, prof in (("major", _MAJ), ("minor", _MIN)):
            r = np.corrcoef(np.roll(prof, pc), chroma)[0, 1]
            if best is None or r > best[0]:
                best = (r, pc, name)
    return {"tonic": theory.note_name(best[1]), "mode": best[2], "corr": round(float(best[0]), 3)}


def generate_layer(plan: dict, index: int, cache_dir: Path, sr: int = 48000) -> dict:
    """Generate (or reuse) the aligned neural layer for a section; returns a record."""
    import soundfile as sf
    from scipy import signal

    s = plan["sections"][index]
    spec = s.get("neural")
    if not spec:
        return {"status": "none"}
    GenerateRequest, get_backend, source = backend_api()
    b0, b1, seconds = layer_region(plan, index)
    key = theory.key_str(s["key"])
    ident = {"spec": spec, "bpm": s["bpm"], "key": key, "seconds": round(seconds, 6), "sr": sr, "v": 1}
    h = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    aligned = cache_dir / f"{h}.aligned.wav"
    meta_path = cache_dir / f"{h}.json"
    if aligned.exists() and meta_path.exists():
        rec = json.loads(meta_path.read_text())
        rec["cached"] = True
        return rec
    rec = {"section": s["id"], "backend": spec["backend"], "interface": source, "start_bar": b0, "end_bar": b1,
           "gain_db": float(spec.get("gain_db", -10.0)), "request": {"prompt": spec.get("prompt", ""), "seconds": seconds,
           "seed": int(spec.get("seed", 0)), "bpm": s["bpm"], "key": key, "mode": spec.get("mode", "generate")}}
    try:
        be = get_backend(spec["backend"])
        ok, reason = be.available()
    except Exception as e:  # unknown backend name etc.
        ok, reason = False, str(e)
    if not ok:
        rec.update(status="skipped", reason=reason)
        return rec
    raw = cache_dir / f"{h}.raw.wav"
    try:
        req = GenerateRequest(prompt=spec.get("prompt", ""), seconds=float(seconds) + 1.0, out=raw,
                              seed=int(spec.get("seed", 0)), bpm=float(s["bpm"]), key=key,
                              mode=spec.get("mode", "generate"), instrumental=True, extra={})
        res = be.generate(req)
        x, in_sr = sf.read(str(res.audio), dtype="float32", always_2d=True)
    except Exception as e:
        rec.update(status="skipped", reason=f"generation failed: {e}")
        return rec
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if in_sr != sr:
        g = np.gcd(int(in_sr), sr)
        x = signal.resample_poly(x, sr // g, int(in_sr) // g, axis=0).astype(np.float32)
    mono = x.mean(axis=1)
    tp = estimate_tempo_phase(mono, sr, float(s["bpm"]))
    kk = estimate_key(mono, sr)
    rec["verification"] = {"tempo": tp, "key_detected": kk,
                           "key_match": theory.pitch_class(kk["tonic"]) == theory.pitch_class(s["key"]["tonic"])}
    if not tp["tempo_ok"]:
        rec.update(status="skipped", reason=f"beat verification failed (estimated {tp['bpm_est']} BPM)")
        return rec
    off = int(round(tp["offset_s"] * sr))
    x = x[off:]
    n = int(round(seconds * sr))
    if len(x) < n:
        x = np.concatenate([x, np.zeros((n - len(x), 2), np.float32)])
    x = x[:n]
    bar = int(round(timing.BEATS_PER_BAR * 60 / float(s["bpm"]) * sr))
    f = min(bar, n // 4)
    x[:f] *= np.linspace(0, 1, f, dtype=np.float32)[:, None]
    x[n - f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
    sf.write(str(aligned), x, sr, subtype="FLOAT")
    rec.update(status="ok", path=str(aligned), sha256=hashlib.sha256(aligned.read_bytes()).hexdigest(),
               sample_rate=sr, model=res.model, result_backend=res.backend,
               provenance=res.provenance, trimmed_seconds=off / sr)
    meta_path.write_text(json.dumps(rec, indent=1, default=str))
    return rec
