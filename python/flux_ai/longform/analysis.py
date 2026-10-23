# SPDX-License-Identifier: GPL-3.0-or-later
"""Audio checks: self-similarity across the set and a sample-level jump detector."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import fft, ndimage, signal

FEAT_SR = 16000


def _mel_bank(n_fft: int, sr: int, n_mels: int = 40, fmin: float = 30, fmax: float = 7600) -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    def hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    pts = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        bank[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return bank


class FeatureExtractor:
    def __init__(self, sr: int = FEAT_SR, n_fft: int = 2048, hop: int = 512):
        self.sr, self.n_fft, self.hop = sr, n_fft, hop
        self.window = np.hanning(n_fft)
        self.mel = _mel_bank(n_fft, sr)
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)
        m = (freqs >= 55) & (freqs <= 4000)
        self.chroma_bins = np.where(m)[0]
        self.chroma_pc = ((np.round(12 * np.log2(freqs[m] / 440.0)) + 9) % 12).astype(int)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mono audio at FEAT_SR -> (chroma (12, T), mfcc (13, T))."""
        if len(x) < self.n_fft:
            x = np.pad(x, (0, self.n_fft - len(x)))
        frames = np.lib.stride_tricks.sliding_window_view(x, self.n_fft)[:: self.hop] * self.window
        spec = np.abs(np.fft.rfft(frames, axis=1)).T  # (F, T)
        chroma = np.zeros((12, spec.shape[1]))
        np.add.at(chroma, self.chroma_pc, spec[self.chroma_bins])
        chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
        mfcc = fft.dct(np.log(self.mel @ (spec ** 2) + 1e-10), axis=0, norm="ortho")[1:14]
        return chroma, mfcc


def _fixed_frames(m: np.ndarray, n: int) -> np.ndarray:
    t_src = np.linspace(0, 1, m.shape[1])
    t_dst = np.linspace(0, 1, n)
    return np.stack([np.interp(t_dst, t_src, row) for row in m])


def self_similarity(audio_path: Path, bar_starts: list[int], window_bars: int = 8, threshold: float = 0.95,
                    frames_per_window: int = 32, top: int = 10) -> dict:
    """Cosine similarity of bar-aligned windows (chroma + MFCC sequences).

    Windows are `window_bars` long and aligned to the plan's bar grid, so tempo
    changes do not matter. Each window's chroma and MFCC sequences are resampled
    to a fixed number of frames, standardised across the whole set, and compared
    by cosine similarity; the score is the mean of the chroma and MFCC cosines.
    Exact or near-exact repeats score close to 1. Pairs closer than two windows
    apart are 'adjacent' and not flagged.
    """
    import soundfile as sf

    fx = FeatureExtractor()
    feats_c, feats_m, times = [], [], []
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        g = np.gcd(sr, FEAT_SR)
        up, down = FEAT_SR // g, sr // g
        for i in range(0, len(bar_starts) - window_bars, window_bars):
            a, b = bar_starts[i], bar_starts[i + window_bars]
            f.seek(a)
            x = f.read(b - a, dtype="float32", always_2d=True).mean(axis=1)
            x = signal.resample_poly(x, up, down)
            c, m = fx(x)
            feats_c.append(_fixed_frames(c, frames_per_window).ravel())
            feats_m.append(_fixed_frames(m, frames_per_window).ravel())
            times.append((a / sr, b / sr))
    n = len(times)
    if n < 3:
        return {"windows": n, "threshold": threshold, "flags": [], "max_nonadjacent": None}

    def cos(F):
        F = np.array(F)
        F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
        return F @ F.T

    S = 0.5 * (cos(feats_c) + cos(feats_m))
    iu = np.triu_indices(n, k=2)
    vals = S[iu]
    order = np.argsort(vals)[::-1]
    flags = [{"a": int(iu[0][k]), "b": int(iu[1][k]), "similarity": round(float(vals[k]), 4),
              "a_time": round(times[iu[0][k]][0], 2), "b_time": round(times[iu[1][k]][0], 2)}
             for k in order if vals[k] > threshold]
    nearest = [{"a": int(iu[0][k]), "b": int(iu[1][k]), "similarity": round(float(vals[k]), 4),
                "a_time": round(times[iu[0][k]][0], 2), "b_time": round(times[iu[1][k]][0], 2)} for k in order[:top]]
    return {
        "method": "bar-aligned windows, mean of standardised chroma and MFCC sequence cosines",
        "window_bars": window_bars, "windows": n, "pairs_checked": int(len(vals)), "threshold": threshold,
        "max_nonadjacent": round(float(vals.max()), 4),
        "percentiles": {p: round(float(np.percentile(vals, p)), 4) for p in (50, 90, 99, 99.9)},
        "flag_count": len(flags), "flags": flags[:200], "most_similar": nearest,
    }


def _shifted_max(d: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """out[n] = max(d[n+lo : n+hi+1]) (offsets may be negative)."""
    size = hi - lo + 1
    c = ndimage.maximum_filter1d(d, size=size, mode="constant", cval=0.0)
    idx = np.arange(len(d)) + lo + size // 2
    out = np.zeros_like(d)
    ok = (idx >= 0) & (idx < len(d))
    out[ok] = c[idx[ok]]
    return out


def find_discontinuities(x: np.ndarray, sr: int, ratio: float = 2.5, floor: float = 0.02,
                         window_ms: float = 50.0, guard_ms: float = 1.0) -> np.ndarray:
    """Sample indices of isolated sample-to-sample jumps (clicks, steps, dropouts).

    A jump is flagged when it is larger than `floor` (full scale = 1) and more
    than `ratio` times the largest jump found 1-50 ms before *and* after it.
    Periodic sharp edges that belong to the sound (bright saw resets, noise in
    a hi-hat, the reset of a 25 Hz saw) recur within that window and are not
    flagged; a step from a bad splice or a dropped buffer is isolated and is.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    d = np.abs(np.diff(x, axis=0)).max(axis=1)
    w = max(4, int(window_ms * 1e-3 * sr))
    g = max(1, int(guard_ms * 1e-3 * sr))
    ref = np.maximum(_shifted_max(d, -w, -g), _shifted_max(d, g, w))
    return np.where((d > floor) & (d > ratio * ref))[0]
