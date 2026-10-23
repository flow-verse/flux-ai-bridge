"""Objective checks of generated audio: tempo and key estimators (needs librosa)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from flux_ai import music

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key_from_chroma(chroma_mean: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
    """Return (best key, correlation, top-3 [(key, r)]) for a 12-bin mean chroma."""
    scores = []
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        for pc in range(12):
            r = float(np.corrcoef(chroma_mean, np.roll(prof, pc))[0, 1])
            scores.append((music.key_name(pc, mode), r))
    scores.sort(key=lambda s: -s[1])
    return scores[0][0], scores[0][1], scores[:3]


def _fit_tempo(beat_times: np.ndarray) -> float:
    """Tempo from the least-squares slope of beat times; robust to a few dropped beats."""
    t = np.asarray(beat_times, dtype=float)
    if len(t) < 8:
        return float("nan")
    period = float(np.median(np.diff(t)))
    # Count beats per interval locally (1 normally, 2 if the tracker skipped one) so a
    # skipped beat does not bend the fit and frame quantisation does not accumulate.
    steps = np.maximum(1, np.round(np.diff(t) / period))
    idx = np.concatenate([[0.0], np.cumsum(steps)])
    slope = np.polyfit(idx, t, 1)[0]
    return float(60.0 / slope)


def analyse(path: Path, target_bpm: float | None = None, target_key: str | None = None) -> dict:
    import librosa
    import scipy.stats

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    dur = len(y) / sr
    hop = 256
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # librosa's global tempo is quantised to the autocorrelation lag grid (about 1.4 BPM
    # near 122 BPM at hop 256), which is too coarse for a +/-1 BPM test. So the beat tracker
    # supplies beat times and the tempo is the least-squares slope of beat time vs beat index.
    tempo_lib, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    tempo_lib = float(np.atleast_1d(tempo_lib)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    tempo_fit = _fit_tempo(beat_times)
    # librosa's default tempo prior is centred on 120 BPM, which flatters a 122 BPM target;
    # also report the estimate with a flat prior over 40-240 BPM.
    tempo_flat = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop, prior=scipy.stats.uniform(40, 200)))[0])
    _, beats_flat = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, bpm=tempo_flat)
    tempo_flat_fit = _fit_tempo(librosa.frames_to_time(beats_flat, sr=sr, hop_length=hop))
    tempo_default = tempo_fit if np.isfinite(tempo_fit) else tempo_lib
    tempo_flat = tempo_flat_fit if np.isfinite(tempo_flat_fit) else tempo_flat
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, r, top = estimate_key_from_chroma(chroma.mean(axis=1))
    out = {
        "file": str(path),
        "seconds": round(dur, 3),
        "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12)), 2),
        "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(y)) + 1e-12)), 2),
        "silence_fraction": round(float(np.mean(np.abs(y) < 1e-3)), 4),
        "bpm_est": round(tempo_default, 2),
        "bpm_est_flat_prior": round(tempo_flat, 2),
        "bpm_librosa_global": round(tempo_lib, 2),
        "beats_tracked": int(len(beat_times)),
        "key_est": key,
        "key_corr": round(r, 3),
        "key_top3": [(k, round(v, 3)) for k, v in top],
    }
    if target_bpm:
        err = tempo_default - target_bpm
        out["bpm_abs_err"] = round(abs(err), 2)
        out["bpm_within_1"] = abs(err) <= 1.0
        out["bpm_within_1_flat_prior"] = abs(tempo_flat - target_bpm) <= 1.0
        out["bpm_within_1_octave_ok"] = any(abs(tempo_default * m - target_bpm) <= 1.0 for m in (1.0, 2.0, 0.5))
    if target_key:
        want = music.normalise_key(target_key)
        out["key_match"] = key == want
        pc, mode = music.parse_key(target_key)
        rel = music.key_name((pc + 3) % 12, "major") if mode == "minor" else music.key_name((pc - 3) % 12, "minor")
        out["key_match_or_relative"] = key in (want, rel)
    return out
