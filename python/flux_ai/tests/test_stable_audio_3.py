"""Stable Audio 3 adapter logic that needs no model: paste-back, spans, runtime choice."""
import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from flux_ai.backends import GenerateRequest  # noqa: E402
from flux_ai.backends import stable_audio_3 as sa3  # noqa: E402

SR = 44100


def _tone(seconds, f=220.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * f * t)
    return np.c_[x, x]


def test_paste_back_keeps_context_exact_outside_the_region(tmp_path):
    ctx = _tone(6.0)
    gen = 0.9 * _tone(6.0) * 1.01  # re-decoded and re-levelled, like the MLX runtime's output
    gen[2 * SR:4 * SR] = _tone(2.0, f=330.0, amp=0.4)  # the regenerated span
    out = tmp_path / "o.wav"
    sf.write(str(out), gen, SR, subtype="PCM_16")
    sa3._paste_back(out, ctx.astype(np.float32), 2.0, 4.0)
    y, _ = sf.read(str(out))
    q = 2 / 32768
    assert np.abs(y[: int(1.9 * SR)] - ctx[: int(1.9 * SR)]).max() <= q
    assert np.abs(y[int(4.1 * SR):] - ctx[int(4.1 * SR):]).max() <= q
    assert np.abs(y[int(2.5 * SR):int(3.5 * SR)] - ctx[int(2.5 * SR):int(3.5 * SR)]).max() > 0.1
    assert np.abs(np.diff(y[:, 0])).max() < 0.1  # crossfades, no steps at the edges


def test_paste_back_for_continue_keeps_the_whole_context(tmp_path):
    ctx = _tone(3.0)
    gen = np.r_[0.8 * ctx, _tone(2.0, f=440.0, amp=0.3)]
    out = tmp_path / "c.wav"
    sf.write(str(out), gen, SR, subtype="PCM_16")
    sa3._paste_back(out, ctx.astype(np.float32), 3.0, 5.0)
    y, _ = sf.read(str(out))
    assert len(y) == 5 * SR
    assert np.abs(y[: int(2.9 * SR)] - ctx[: int(2.9 * SR)]).max() <= 2 / 32768


def test_span_limits_and_modes(tmp_path):
    b = sa3.StableAudio3Backend()
    req = lambda **kw: GenerateRequest(prompt="x", seconds=kw.pop("seconds", 30), out=tmp_path / "o.wav", **kw)  # noqa: E731
    assert b._span(req(), "small-music", None) == (30.0, None, None)
    assert b._span(req(mode="continue", seconds=20), "small-music", 30.0) == (50.0, 30.0, 50.0)
    assert b._span(req(mode="repaint", extra={"start": 5, "end": 8}), "small-music", 30.0) == (30.0, 5.0, 8.0)
    with pytest.raises(ValueError, match="at most 120"):
        b._span(req(seconds=121), "small-music", None)
    assert b._span(req(seconds=380), "medium", None)[0] == 380.0


def test_runtime_choice_and_pins(monkeypatch, tmp_path):
    monkeypatch.setattr(sa3.sys, "platform", "darwin")
    monkeypatch.setattr(sa3.platform, "machine", lambda: "arm64")
    assert sa3.default_runtime() == "mlx"
    monkeypatch.setattr(sa3.sys, "platform", "linux")
    assert sa3.default_runtime() == "torch"
    assert len(sa3.MLX_COMMIT) == 40 and sa3.MLX_COMMIT[:12] in str(sa3.mlx_runtime_dir())
    monkeypatch.setenv("FLUX_SA3_MLX_DIR", str(tmp_path / "rt"))
    assert sa3.mlx_runtime_dir() == tmp_path / "rt"
    with pytest.raises(RuntimeError, match="no optimized/mlx"):
        sa3.ensure_mlx_runtime()
    assert set(sa3.MLX_TIERS) == set(sa3.TIERS) == set(sa3.MAX_SECONDS)
