# SPDX-License-Identifier: GPL-3.0-or-later
"""Small numpy/scipy synth used by the built-in renderer.

PLACEHOLDER SOUND: this exists so the long-form pipeline runs with no DAW or
plugins. It is not meant to compete with flux-host + Surge XT rendering, which
replaces it once `flux-host render` (phase P1) exists.

All voices have attack/release ramps so nothing starts or stops with a step.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def db(x: float) -> float:
    return float(10 ** (x / 20))


def rbj_lowpass(fc: float, q: float, sr: int):
    fc = min(max(fc, 20.0), 0.45 * sr)
    w = 2 * np.pi * fc / sr
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def rbj_highpass(fc: float, q: float, sr: int):
    fc = min(max(fc, 20.0), 0.45 * sr)
    w = 2 * np.pi * fc / sr
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def rbj_bandpass(fc: float, q: float, sr: int):
    fc = min(max(fc, 20.0), 0.45 * sr)
    w = 2 * np.pi * fc / sr
    alpha = np.sin(w) / (2 * q)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w), 1 - alpha])
    return b / a[0], a / a[0]


def lp(x, fc, sr, q=0.707):
    return signal.lfilter(*rbj_lowpass(fc, q, sr), x)


def hp(x, fc, sr, q=0.707):
    return signal.lfilter(*rbj_highpass(fc, q, sr), x)


def bp(x, fc, sr, q=1.0):
    return signal.lfilter(*rbj_bandpass(fc, q, sr), x)


def edge_ramps(x: np.ndarray, sr: int, attack_s: float = 0.0005, release_s: float = 0.003) -> np.ndarray:
    n = len(x)
    a = min(n, max(1, int(attack_s * sr)))
    r = min(n, max(1, int(release_s * sr)))
    x[:a] *= np.linspace(0, 1, a, endpoint=False)
    x[n - r:] *= np.linspace(1, 0, r)
    return x


# ------------------------------------------------------------------ oscillators

def _phase(freq, n: int, sr: int, phase0: float = 0.0) -> np.ndarray:
    if np.isscalar(freq):
        return (phase0 + np.arange(n) * (freq / sr)) % 1.0
    return (phase0 + np.cumsum(freq / sr)) % 1.0


def _blep(t: np.ndarray, dt) -> np.ndarray:
    out = np.zeros_like(t)
    dt = np.broadcast_to(dt, t.shape)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1
    m = t > 1 - dt
    x = (t[m] - 1) / dt[m]
    out[m] = x * x + x + x + 1
    return out


def saw(freq, n, sr, phase0=0.0):
    ph = _phase(freq, n, sr, phase0)
    dt = (freq / sr) if np.isscalar(freq) else freq / sr
    return 2 * ph - 1 - _blep(ph, dt)


def square(freq, n, sr, phase0=0.0):
    return 0.5 * (saw(freq, n, sr, phase0) - saw(freq, n, sr, (phase0 + 0.5) % 1.0))


def sine(freq, n, sr, phase0=0.0):
    return np.sin(2 * np.pi * _phase(freq, n, sr, phase0))


# ------------------------------------------------------------------ envelopes

def adsr(n_on: int, n_total: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    t = np.arange(n_total) / sr
    a = max(a, 0.001)
    env = np.interp(t, [0, a, a + max(d, 0.001)], [0, 1, s], right=s)
    t_off = n_on / sr
    lvl_off = float(np.interp(t_off, [0, a, a + max(d, 0.001)], [0, 1, s], right=s))
    rel = t >= t_off
    r = max(r, 0.003)
    env[rel] = lvl_off * np.clip(1 - (t[rel] - t_off) / r, 0, 1)
    return env


# ------------------------------------------------------------------ drums

def drum_oneshot(kind: str, p: dict, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Mono one-shot for a drum instrument (always starts and ends at zero)."""
    level = db(p.get("level_db", -12))
    k = p.get("kind", kind)
    if kind == "kick" or k == "doum":
        dec = p.get("decay", 0.4)
        n = int(sr * dec * 5)
        t = np.arange(n) / sr
        f0 = p.get("freq", 50)
        f = f0 * (1 + p.get("pitch_amount", 3) * np.exp(-t / p.get("pitch_decay", 0.04)))
        body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / dec)
        # deterministic tonal click (a noise click can produce isolated single-sample spikes)
        click = np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.003) * np.clip(t / 0.001, 0, 1) * p.get("click", 0.2)
        x = body + click
        if k == "doum":
            x += 0.3 * bp(rng.standard_normal(n), 300, sr, 2) * np.exp(-t / 0.02)
    elif k == "snare":
        dec = p.get("decay", 0.2)
        n = int(sr * dec * 5)
        t = np.arange(n) / sr
        x = 0.6 * bp(rng.standard_normal(n), p.get("tone", 1800), sr, 0.7) * np.exp(-t / dec)
        x += 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.06)
    elif kind == "clap":
        if k == "tek":
            dec = p.get("decay", 0.05)
            n = int(sr * max(dec * 6, 0.05))
            t = np.arange(n) / sr
            x = bp(rng.standard_normal(n), p.get("tone", 2500), sr, 1.5) * np.exp(-t / dec)
            x += 0.4 * np.sin(2 * np.pi * 620 * t) * np.exp(-t / 0.02)
        else:
            dec = p.get("decay", 0.18)
            n = int(sr * (0.03 + dec * 5))
            t = np.arange(n) / sr
            env = np.zeros(n)
            for off in (0.0, 0.011, 0.022):
                m = t >= off
                env[m] += np.exp(-(t[m] - off) / 0.006)
            m = t >= 0.03
            env[m] += 0.6 * np.exp(-(t[m] - 0.03) / dec)
            x = bp(rng.standard_normal(n), p.get("tone", 1500), sr, 1.2) * env * 1.5
    elif kind in ("hat_closed", "hat_open", "ride"):
        dec = p.get("decay", 0.05)
        n = int(sr * max(dec * 6, 0.03))
        t = np.arange(n) / sr
        noise = hp(hp(rng.standard_normal(n), p.get("tone", 8000), sr), p.get("tone", 8000) * 0.8, sr)
        metal = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in (317, 461, 587, 723, 853, 1011))
        metal = hp(metal, 6000, sr) * 0.08
        x = (noise + metal) * np.exp(-t / dec)
        if k == "riq":
            x += 0.5 * bp(rng.standard_normal(n), 5000, sr, 3) * np.exp(-t / (dec * 1.5))
    elif kind == "crash":
        dec = 1.2
        n = int(sr * dec * 4)
        t = np.arange(n) / sr
        x = hp(rng.standard_normal(n), 5000, sr) * np.exp(-t / dec) * 0.6
        level = db(-26)
    else:  # perc
        dec = p.get("decay", 0.05)
        if k == "bell":
            n = int(sr * dec * 4)
            t = np.arange(n) / sr
            f = p.get("freq", 1200)
            x = np.sin(2 * np.pi * f * t + 2.0 * np.exp(-t / 0.3) * np.sin(2 * np.pi * f * 3.5 * t)) * np.exp(-t / dec)
        else:
            n = int(sr * max(dec * 6, 0.04))
            t = np.arange(n) / sr
            f = p.get("freq", 900)
            x = 0.7 * np.sin(2 * np.pi * f * t) * np.exp(-t / dec) + 0.5 * bp(rng.standard_normal(n), f * 1.8, sr, 2) * np.exp(-t / (dec * 0.5))
    x = x / (np.max(np.abs(x)) + 1e-9) * level
    return edge_ramps(x.astype(np.float32), sr, 0.0005, 0.004)


# ------------------------------------------------------------------ melodic voices

def voice(p: dict, freq: float, dur_s: float, vel: float, bright: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """One melodic note, mono, including its release tail."""
    wave = p.get("wave", "saw")
    rel = float(p.get("release", 0.1))
    n_on = max(1, int(dur_s * sr))
    n = n_on + int(rel * sr) + 1
    t = np.arange(n) / sr
    vib = float(p.get("vibrato_cents", 0) or 0)
    f = freq
    if vib:
        depth = (2 ** (vib / 1200) - 1) * np.clip((t - 0.12) / 0.2, 0, 1)
        f = freq * (1 + depth * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6.28)))
    if wave == "sine":
        x = sine(f, n, sr, rng.uniform())
    elif wave == "fm":
        ratio, index = p.get("fm_ratio", 2.0), p.get("fm_index", 1.0)
        ienv = index * (0.4 + 0.6 * np.exp(-t / max(p.get("env_decay", 0.3), 0.01))) * (0.6 + 0.6 * bright)
        mod = np.sin(2 * np.pi * _phase(f * ratio if np.isscalar(f) else f * ratio, n, sr))
        x = np.sin(2 * np.pi * _phase(f, n, sr) + ienv * mod)
    else:
        voices = max(1, int(p.get("voices", 1)))
        det = float(p.get("detune_cents", 0) or 0)
        osc = square if wave == "square" else saw
        x = np.zeros(n)
        for v in range(voices):
            cents = 0.0 if voices == 1 else det * (2 * v / (voices - 1) - 1)
            x += osc(f * 2 ** (cents / 1200), n, sr, rng.uniform())
        x /= np.sqrt(voices)
    sub = float(p.get("sub", 0) or 0)
    if sub:
        x = x + sub * sine(f / 2 if np.isscalar(f) else f / 2, n, sr)
    if wave not in ("sine", "fm"):
        cutoff = float(p.get("cutoff", 1500)) * (0.45 + 1.3 * bright) * (0.75 + 0.35 * vel)
        dark = lp(x, cutoff, sr, 0.9)
        amt = float(p.get("env_amount", 0) or 0)
        if amt > 0:
            light = lp(x, cutoff * (1 + amt), sr, 0.9)
            fenv = np.exp(-t / max(p.get("env_decay", 0.2), 0.005))
            x = dark + fenv * (light - dark)
        else:
            x = dark
    if wave == "pluck":
        env = np.exp(-t / max(p.get("decay", 0.4), 0.01)) * np.clip(t / 0.002, 0, 1)
        env *= np.clip(1 - (t - n_on / sr) / max(rel, 0.003), 0, 1) ** (t >= n_on / sr)
    else:
        env = adsr(n_on, n, sr, p.get("attack", 0.005), p.get("decay", 0.2), p.get("sustain", 0.6), rel)
    y = (x * env * vel).astype(np.float32)
    return edge_ramps(y, sr, 0.0005, 0.002)


def midi_to_hz(p: float) -> float:
    return 440.0 * 2 ** ((p - 69) / 12)


def pan_gains(pan: float) -> tuple[float, float]:
    a = (pan + 1) * np.pi / 4
    return float(np.cos(a)), float(np.sin(a))


def add_at(buf: np.ndarray, x: np.ndarray, start: int, gl: float, gr: float) -> None:
    if start >= len(buf) or start + len(x) <= 0:
        return
    if start < 0:
        x = x[-start:]
        start = 0
    end = min(len(buf), start + len(x))
    seg = x[: end - start]
    buf[start:end, 0] += seg * gl
    buf[start:end, 1] += seg * gr


# ------------------------------------------------------------------ effects

def reverb_ir(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = np.exp(-6.9 * t / seconds)
    ir = np.stack([lp(rng.standard_normal(n), 6000, sr) * env, lp(rng.standard_normal(n), 6000, sr) * env], axis=1)
    pre = int(0.012 * sr)
    ir = np.concatenate([np.zeros((pre, 2)), ir])
    ir[pre:pre + int(0.004 * sr)] *= np.linspace(0, 1, int(0.004 * sr))[:, None]
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir.astype(np.float32)


def convolve_stereo(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for c in range(2):
        out[:, c] = signal.oaconvolve(x[:, c], ir[:, c], mode="full")[: len(x)]
    return out


def pingpong(x: np.ndarray, delay_samples: int, feedback: float = 0.38, taps: int = 4, sr: int = 48000) -> np.ndarray:
    out = np.zeros_like(x)
    mono = x.mean(axis=1)
    mono = lp(hp(mono, 300, sr), 4000, sr).astype(np.float32)
    g = 1.0
    for k in range(1, taps + 1):
        g *= feedback
        d = k * delay_samples
        if d >= len(x):
            break
        ch = (k - 1) % 2
        out[d:, ch] += mono[: len(x) - d] * g
    return out


def duck_envelope(onsets: np.ndarray, vels: np.ndarray, n: int, sr: int, release: float, depth: float) -> np.ndarray:
    """Sidechain-style gain (1 - depth * smooth pulse after each kick)."""
    if depth <= 0 or len(onsets) == 0:
        return np.ones(n, dtype=np.float32)
    imp = np.zeros(n)
    ok = onsets < n
    np.add.at(imp, onsets[ok], vels[ok])
    slow = signal.lfilter([1.0], [1.0, -np.exp(-1 / (release * sr))], imp)
    fast = signal.lfilter([1.0], [1.0, -np.exp(-1 / (0.004 * sr))], imp)
    pulse = slow - fast
    norm = np.max(signal.lfilter([1.0], [1.0, -np.exp(-1 / (release * sr))], np.r_[1.0, np.zeros(int(sr * 0.05))])
                  - signal.lfilter([1.0], [1.0, -np.exp(-1 / (0.004 * sr))], np.r_[1.0, np.zeros(int(sr * 0.05))]))
    return (1 - depth * np.clip(pulse / norm, 0, 1)).astype(np.float32)


def texture(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, 2), dtype=np.float32)
    if kind == "vinyl":
        hiss = lp(rng.standard_normal(n), 5000, sr) * db(-58)
        crackle = np.zeros(n)
        k = rng.poisson(6 * n / sr)
        pos = rng.integers(0, n, k)
        crackle[pos] = rng.uniform(-1, 1, k) * db(-38)
        crackle = bp(crackle, 3000, sr, 0.8)
        out += (hiss + crackle)[:, None].astype(np.float32)
    elif kind == "air":
        t = np.arange(n) / sr
        for c in range(2):
            mod = 0.6 + 0.4 * np.sin(2 * np.pi * t / rng.uniform(9, 17) + rng.uniform(0, 6))
            out[:, c] = (lp(rng.standard_normal(n), 1800, sr) * mod * db(-44)).astype(np.float32)
    return out
