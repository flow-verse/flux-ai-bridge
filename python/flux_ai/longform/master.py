# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-mix mastering with bounded memory.

Pass 1 streams the assembled mix through a BS.1770 loudness meter (the same
K-weighting coefficients pyloudnorm uses, filter state carried across chunks).
Pass 2 applies the normalisation gain and a look-ahead true-peak limiter
(4x oversampled peak estimate), measures the result again while streaming it
to the output file, and repeats with a corrected gain if the measured loudness
misses the target by more than `retune_db`. Memory is O(chunk), not O(set).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage, signal


class StreamingLoudness:
    """Integrated loudness (ITU-R BS.1770-4 gating) over streamed chunks."""

    def __init__(self, sr: int, channels: int = 2):
        import pyloudnorm

        meter = pyloudnorm.Meter(sr)
        self.stages = [(f.passband_gain, f.b, f.a) for f in meter._filters.values()]
        self.zi = [[signal.lfilter_zi(b, a) * 0.0 for _ in range(channels)] for _, b, a in self.stages]
        self.sr = sr
        self.seg = int(round(0.1 * sr))
        self.channels = channels
        self.acc = np.zeros(channels)
        self.fill = 0
        self.segs: list[np.ndarray] = []
        self.samples = 0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        for c in range(self.channels):
            v = x[:, c]
            for si, (g, b, a) in enumerate(self.stages):
                v, self.zi[si][c] = signal.lfilter(b, a, v, zi=self.zi[si][c])
                v = g * v
            y[:, c] = v
        self.samples += len(y)
        sq = y * y
        pos = 0
        while pos < len(sq):
            take = min(self.seg - self.fill, len(sq) - pos)
            self.acc += sq[pos:pos + take].sum(axis=0)
            self.fill += take
            pos += take
            if self.fill == self.seg:
                self.segs.append(self.acc.copy())
                self.acc[:] = 0
                self.fill = 0

    def integrated(self) -> float:
        if len(self.segs) < 4:
            return float("-inf")
        s = np.array(self.segs)  # (nseg, ch) energy per 100 ms
        z = (s[:-3] + s[1:-2] + s[2:-1] + s[3:]) / (4 * self.seg)  # 400 ms blocks, 75 % overlap
        G = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[: self.channels]
        with np.errstate(divide="ignore"):
            lj = -0.691 + 10 * np.log10((z * G).sum(axis=1))
        abs_ok = lj >= -70.0
        if not np.any(abs_ok):
            return float("-inf")
        gamma_r = -0.691 + 10 * np.log10((z[abs_ok].mean(axis=0) * G).sum()) - 10.0
        ok = abs_ok & (lj > gamma_r)
        return float(-0.691 + 10 * np.log10((z[ok].mean(axis=0) * G).sum()))


class StreamingTruePeak:
    """4x oversampled sample-peak estimate (BS.1770 Annex 2 style) across chunks."""

    CTX = 64

    def __init__(self, channels: int = 2):
        # `tail` = already-scanned context followed by `unscanned` samples that still
        # lacked right-hand context for the resampling filter.
        self.tail = np.zeros((self.CTX, channels))
        self.unscanned = 0
        self.peak = 0.0

    def _scan(self, buf: np.ndarray, lo: int, hi: int) -> None:
        if hi <= lo:
            return
        up = signal.resample_poly(buf, 4, 1, axis=0)
        self.peak = max(self.peak, float(np.max(np.abs(up[4 * lo:4 * hi]))))

    def update(self, x: np.ndarray) -> None:
        buf = np.concatenate([self.tail, np.asarray(x, dtype=np.float64)])
        lo = len(self.tail) - self.unscanned
        hi = len(buf) - self.CTX
        self._scan(buf, lo, hi)
        done = max(lo, hi)
        self.unscanned = len(buf) - done
        self.tail = buf[max(0, done - self.CTX):]

    def finish(self) -> float:
        buf = np.concatenate([self.tail, np.zeros((self.CTX, self.tail.shape[1]))])
        self._scan(buf, len(self.tail) - self.unscanned, len(self.tail))
        self.unscanned = 0
        return self.peak

    @property
    def dbtp(self) -> float:
        return 20 * np.log10(max(self.peak, 1e-12))


def true_peak_dbtp(x: np.ndarray) -> float:
    up = signal.resample_poly(np.asarray(x, dtype=np.float64), 4, 1, axis=0)
    return float(20 * np.log10(max(np.max(np.abs(up)), 1e-12)))


def _window_reduce(fn, x: np.ndarray, before: int, after: int) -> np.ndarray:
    """out[n] = fn(x[n-before : n+after+1]) using scipy's centred filters."""
    size = before + after + 1
    c = fn(x, size=size, mode="nearest")
    idx = np.clip(np.arange(len(x)) - before + size // 2, 0, len(x) - 1)
    return c[idx]


def limiter_gain(x: np.ndarray, ceiling: float, lookahead: int, hold: int) -> np.ndarray:
    """Smooth gain g[n] <= ceiling / truepeak[n] for every sample (no overshoot by construction)."""
    up = np.abs(signal.resample_poly(x, 4, 1, axis=0))
    peak = up.reshape(len(x), 4, -1).max(axis=(1, 2))
    g_req = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    m = _window_reduce(ndimage.minimum_filter1d, g_req, hold, lookahead - 1)
    return _window_reduce(ndimage.uniform_filter1d, m, lookahead - 1, 0)


@dataclass
class MasterResult:
    gain_db: float
    premaster_lufs: float
    lufs: float
    true_peak_dbtp: float
    limiter_max_reduction_db: float
    passes: int
    frames: int


def master(read: Callable[[int, int], np.ndarray], total: int, sr: int, writer_factory: Callable,
           target_lufs: float = -14.0, tolerance: float = 0.5, tp_dbtp: float = -1.0,
           chunk: int = 1 << 18, retune_db: float = 0.1, progress: Callable | None = None) -> MasterResult:
    """Normalise + limit the stream `read(a, b)` of length `total` into `writer_factory()`."""
    meter = StreamingLoudness(sr)
    for a in range(0, total, chunk):
        meter.update(read(a, min(total, a + chunk)))
        if progress:
            progress(0.3 * min(1.0, (a + chunk) / total))
    pre = meter.integrated()
    gain_db = target_lufs - pre
    ceiling_db = tp_dbtp - 0.3
    look, hold = int(0.005 * sr), int(0.040 * sr)
    ctx = hold + look + 256
    for p in range(1, 5):
        g = 10 ** (gain_db / 20)
        ceil = 10 ** (ceiling_db / 20)
        out_meter, tp = StreamingLoudness(sr), StreamingTruePeak()
        max_red = 0.0
        with writer_factory() as w:
            for a in range(0, total, chunk):
                b = min(total, a + chunk)
                a0, b0 = max(0, a - ctx), min(total, b + ctx)
                x = read(a0, b0) * g
                gl = limiter_gain(x, ceil, look, hold)
                y = (x * gl[:, None])[a - a0:a - a0 + (b - a)]
                max_red = max(max_red, float(-20 * np.log10(max(np.min(gl[a - a0:a - a0 + (b - a)]), 1e-9))))
                out_meter.update(y)
                tp.update(y)
                w.write(y.astype(np.float32))
                if progress:
                    progress(0.3 + 0.7 * b / total)
        lufs = out_meter.integrated()
        peak = 20 * np.log10(max(tp.finish(), 1e-12))
        miss = target_lufs - lufs
        if abs(miss) <= retune_db and peak <= tp_dbtp:
            break
        if peak > tp_dbtp:
            ceiling_db -= (peak - tp_dbtp) + 0.05
        if abs(miss) > retune_db:
            gain_db += miss
    return MasterResult(round(gain_db, 3), round(pre, 3), round(lufs, 3), round(peak, 3), round(max_red, 2), p, total)
