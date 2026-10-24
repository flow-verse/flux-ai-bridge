"""Stable Audio 3 adapter (opt-in; Stability AI Community License, USD 1M revenue cap).

Two runtimes, chosen with ``extra["runtime"]`` (default: ``mlx`` on Apple
Silicon, ``torch`` elsewhere):

``mlx``
    Stability's own MLX runtime (``optimized/mlx`` in
    github.com/Stability-AI/stable-audio-3, MIT). Flux fetches it once, at a
    pinned commit, into ``~/.cache/flux-ai/runtimes/`` (or ``$FLUX_SA3_MLX_DIR``)
    and runs its ``scripts/sa3_mlx.py`` with this interpreter, which needs the
    ``[stable-audio-3-mlx]`` extra (mlx, sentencepiece, huggingface_hub,
    soundfile). Weights come from the ungated ``stabilityai/stable-audio-3-optimized``
    repo on first use; the licence still applies, so Flux still requires
    ``--accept-licence stable-audio-3``.

``torch``
    The ``stable-audio-3`` PyTorch library. It pins torch==2.7.1 and
    transformers>=5.8, which conflict with ACE-Step 1.5, so install it in its
    own venv (``uv pip install -p .venv-sa3 -e "python[stable-audio-3]"``). Its
    weights are gated: accept the terms on the model pages and ``hf auth login``.

Models (``extra["model"]``): ``small-music`` (default, 568M, up to 120 s),
``medium`` (2.3B, music and sound effects, up to 380 s) and, MLX only,
``small-sfx`` (sound effects, up to 120 s). Modes: generate, continue (output =
context + continuation), repaint/infill (``extra["start"]``/``extra["end"]``).
There is no BPM/key input; they are appended to the prompt ("…, 122 BPM, A minor").
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from flux_ai import licences, provenance
from flux_ai.backends import GenerateRequest, GenerateResult, check_request, finalize, report

TIERS = {"small-music": "small-music", "medium": "medium", "small-sfx": "small-sfx"}
HF_REPOS = {"small-music": "stabilityai/stable-audio-3-small-music", "medium": "stabilityai/stable-audio-3-medium",
            "small-sfx": "stabilityai/stable-audio-3-small-sfx"}
MAX_SECONDS = {"small-music": 120.0, "medium": 380.0, "small-sfx": 120.0}

# MLX runtime: Stability's code at a pinned commit, and its names for each tier
MLX_REPO = "https://github.com/Stability-AI/stable-audio-3"
MLX_COMMIT = "77e0398c26f7d22d801ec1bb036307e8b231f741"
MLX_WEIGHTS_REPO = "stabilityai/stable-audio-3-optimized"
MLX_TIERS = {"small-music": ("sm-music", "same-s"), "medium": ("medium", "same-l"), "small-sfx": ("sm-sfx", "same-s")}
MLX_SAMPLE_RATE = 44100


def default_runtime() -> str:
    return "mlx" if sys.platform == "darwin" and platform.machine() == "arm64" else "torch"


def mlx_runtime_dir() -> Path:
    """Stability's MLX runtime checkout (``optimized/mlx`` inside it)."""
    env = os.environ.get("FLUX_SA3_MLX_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "flux-ai" / "runtimes" / f"stable-audio-3-{MLX_COMMIT[:12]}"


def ensure_mlx_runtime(progress: Callable[[float, str], None] | None = None) -> Path:
    """Fetch Stability's runtime at MLX_COMMIT if missing; returns its optimized/mlx directory."""
    root = mlx_runtime_dir()
    script = root / "optimized" / "mlx" / "scripts" / "sa3_mlx.py"
    if script.is_file():
        return script.parent.parent
    if os.environ.get("FLUX_SA3_MLX_DIR"):
        raise RuntimeError(f"$FLUX_SA3_MLX_DIR={root} has no optimized/mlx/scripts/sa3_mlx.py")
    if shutil.which("git") is None:
        raise RuntimeError("git is needed to fetch Stability's MLX runtime")
    report(progress, 0.01, f"fetching Stability's MLX runtime ({MLX_COMMIT[:12]})")
    tmp = root.with_name(root.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True, text=True)  # noqa: E731
    run("init", "-q", str(tmp))
    run("-C", str(tmp), "fetch", "-q", "--depth", "1", MLX_REPO, MLX_COMMIT)
    run("-C", str(tmp), "checkout", "-q", "FETCH_HEAD")
    tmp.rename(root)
    return root / "optimized" / "mlx"


class StableAudio3Backend:
    name = "stable-audio-3"
    licence_id = "stable-audio-3"
    modes = frozenset({"generate", "continue", "repaint", "infill"})
    models = list(TIERS)
    output_kind = "audio"

    def __init__(self):
        self._models: dict[str, object] = {}

    # ------------------------------------------------------------------ availability
    def available(self) -> tuple[bool, str]:
        rt = default_runtime()
        if rt == "mlx":
            missing = [m for m in ("mlx", "sentencepiece", "huggingface_hub", "soundfile")
                       if importlib.util.find_spec(m) is None]
            if missing:
                return False, ('MLX runtime needs: uv pip install -e "python[stable-audio-3-mlx]" '
                               f"(missing {', '.join(missing)})")
            state = "runtime fetched" if (mlx_runtime_dir() / "optimized").is_dir() else \
                "runtime fetched on first use"
            where = f"MLX ({state}); weights from {MLX_WEIGHTS_REPO} on first use"
        else:
            if importlib.util.find_spec("stable_audio_3") is None:
                return False, (
                    "stable-audio-3 not installed. It pins torch==2.7.1 and transformers>=5.8, which conflict "
                    'with ace-step, so install it in its own venv: uv pip install -p .venv-sa3 -e '
                    '"python[stable-audio-3]". Weights are gated on Hugging Face: accept the terms at '
                    "https://huggingface.co/stabilityai/stable-audio-3-small-music and run `hf auth login`.")
            where = "PyTorch; weights download on first use (HF gate access required)"
        if not licences.is_accepted(self.licence_id):
            return True, f"{where}; opt-in: needs --accept-licence {self.licence_id} (USD 1M revenue cap)"
        return True, where

    # ------------------------------------------------------------------ generation
    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult:
        check_request(self, req)
        licences.require(self.licence_id)
        tier = str(req.extra.get("model", "small-music"))
        if tier not in TIERS:
            raise ValueError(f"unknown stable-audio-3 model '{tier}'; choose from {', '.join(TIERS)}")
        runtime = str(req.extra.get("runtime", default_runtime()))
        if runtime not in ("mlx", "torch"):
            raise ValueError("extra runtime must be 'mlx' or 'torch'")
        prompt = req.prompt + "".join(
            s for s in ((f", {req.bpm:g} BPM" if req.bpm else ""), (f", {req.key}" if req.key else "")))
        if runtime == "mlx":
            return self._generate_mlx(req, tier, prompt, progress)
        if tier == "small-sfx":
            raise ValueError("small-sfx is wired for the MLX runtime only")
        return self._generate_torch(req, tier, prompt, progress)

    def _span(self, req: GenerateRequest, tier: str, ctx_s: float | None) -> tuple[float, float | None, float | None]:
        """(total seconds, inpaint start, inpaint end) for the request's mode."""
        if req.mode == "generate":
            total, start, end = float(req.seconds), None, None
        elif req.mode == "continue":
            total, start, end = ctx_s + req.seconds, ctx_s, ctx_s + req.seconds
        else:
            start = float(req.extra.get("start", 0.0))
            end = float(req.extra.get("end", min(ctx_s, start + req.seconds)))
            total = ctx_s
        if total > MAX_SECONDS[tier] + 1e-6:
            raise ValueError(f"{tier} generates at most {MAX_SECONDS[tier]:.0f} s in one call (asked {total:.1f} s)")
        return total, start, end

    def _generate_mlx(self, req, tier, prompt, progress) -> GenerateResult:
        import numpy as np
        import soundfile as sf

        started = provenance.now()
        mlx_dir = ensure_mlx_runtime(progress)
        dit, decoder = MLX_TIERS[tier]
        steps, cfg = int(req.extra.get("steps", 8)), float(req.extra.get("cfg_scale", 1.0))
        out = Path(req.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="flux-sa3-") as tmp:
            init = None
            ctx_s = None
            if req.mode != "generate":
                x, sr_in = sf.read(str(req.context_audio), dtype="float32", always_2d=True)
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                x = x[:, :2]
                if sr_in != MLX_SAMPLE_RATE:
                    from scipy import signal

                    g = np.gcd(int(sr_in), MLX_SAMPLE_RATE)
                    x = signal.resample_poly(x, MLX_SAMPLE_RATE // g, int(sr_in) // g, axis=0).astype(np.float32)
                ctx_s = len(x) / MLX_SAMPLE_RATE
            total, start, end = self._span(req, tier, ctx_s)
            if ctx_s is not None:
                n = int(round(total * MLX_SAMPLE_RATE))
                x = np.concatenate([x, np.zeros((max(0, n - len(x)), 2), np.float32)])[:n]
                init = Path(tmp) / "init.wav"
                sf.write(str(init), np.clip(x, -1, 1), MLX_SAMPLE_RATE, subtype="PCM_16")
            cmd = [sys.executable, "scripts/sa3_mlx.py", "--prompt", prompt, "--dit", dit, "--decoder", decoder,
                   "--seconds", f"{total:.6f}", "--steps", str(steps), "--seed", str(int(req.seed)),
                   "--cfg", str(cfg), "--out", str(out)]
            if init is not None:
                cmd += ["--init-audio", str(init), "--inpaint-range", f"{start:.6f},{end:.6f}"]
            report(progress, 0.05, f"Stable Audio 3 {tier} (MLX): loading and sampling")
            env = dict(os.environ, NO_COLOR="1", TERM="dumb", PYTHONUNBUFFERED="1")
            proc = subprocess.run(cmd, cwd=str(mlx_dir), stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, env=env)
            if proc.returncode != 0 or not out.is_file():
                tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-12:])
                if "401" in tail or "403" in tail or "gated" in tail.lower():
                    tail += (f"\nHugging Face refused the weights; check `hf auth whoami` and access to "
                             f"{MLX_WEIGHTS_REPO}.")
                raise RuntimeError(f"Stable Audio 3 MLX runtime failed (exit {proc.returncode}):\n{tail}")
            if init is not None:
                _paste_back(out, x, start, end)
        info = sf.info(str(out))
        report(progress, 1.0, "done")
        return finalize(
            self, req, output=out, model=f"{MLX_WEIGHTS_REPO} ({dit}, {decoder})",
            model_version=f"stable-audio-3 MLX runtime {MLX_COMMIT[:12]}",
            sample_rate=int(info.samplerate), seconds=float(info.duration), started_at=started,
            backend_params={"runtime": "mlx", "prompt": prompt, "dit": dit, "decoder": decoder, "seconds": total,
                            "steps": steps, "cfg_scale": cfg, "seed": int(req.seed),
                            "inpaint_range": None if start is None else [start, end]},
        )

    def _generate_torch(self, req, tier, prompt, progress) -> GenerateResult:
        started = provenance.now()
        import torch
        from stable_audio_3 import StableAudioModel

        if tier not in self._models:
            report(progress, 0.02, f"loading Stable Audio 3 {tier} (gated HF repo {HF_REPOS[tier]})")
            try:
                self._models[tier] = StableAudioModel.from_pretrained(TIERS[tier])
            except Exception as e:
                if type(e).__name__ == "GatedRepoError" or "gated repo" in str(e).lower():
                    raise RuntimeError(
                        f"Hugging Face refused {HF_REPOS[tier]} (gated). Accept Stability AI's terms on "
                        f"https://huggingface.co/{HF_REPOS[tier]} with your account, then `hf auth login`."
                    ) from e
                raise
        model = self._models[tier]
        kw = dict(prompt=prompt, duration=float(req.seconds), seed=int(req.seed),
                  steps=int(req.extra.get("steps", 8)), cfg_scale=float(req.extra.get("cfg_scale", 1.0)))
        if req.mode != "generate":
            import torchaudio

            wav, sr_in = torchaudio.load(str(req.context_audio))
            total, start, end = self._span(req, tier, wav.shape[-1] / sr_in)
            kw.update(inpaint_audio=(sr_in, wav), inpaint_mask_start_seconds=start, inpaint_mask_end_seconds=end,
                      duration=total)
        else:
            self._span(req, tier, None)
        report(progress, 0.1, "sampling")
        out = model.generate(**kw)
        audio = out[0] if out.dim() == 3 else out
        sr = int(model.model_config.get("sample_rate", 44100))
        x = audio.detach().to(torch.float32).cpu().numpy().T
        from flux_ai.music import write_wav

        write_wav(req.out, x, sr)
        report(progress, 1.0, "done")
        return finalize(
            self, req, output=req.out, model=HF_REPOS[tier], model_version="stable-audio-3 " + _version(),
            sample_rate=sr, seconds=x.shape[0] / sr, started_at=started,
            backend_params={"runtime": "torch", **{k: v for k, v in kw.items() if k not in ("inpaint_audio",)}},
        )


def _paste_back(out: Path, context, start: float, end: float, fade_s: float = 0.05) -> None:
    """Keep the context sample-exact outside [start, end).

    The MLX runtime re-decodes the whole clip and peak-normalises it, so even
    the kept audio comes back slightly changed. Match the new clip's level to
    the context over the kept part, then splice the original back in with
    short equal-power crossfades at the region edges.
    """
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(out), dtype="float64", always_2d=True)
    x = np.asarray(context, dtype=np.float64)[: len(y)]
    n = len(y)
    a, b = int(round(start * sr)), min(n, int(round(end * sr)))
    keep = np.ones(n, bool)
    keep[a:b] = False
    keep[len(x):] = False
    if keep.any():
        den = float((y[keep] ** 2).sum())
        if den > 0:
            y *= float((x[keep[: len(x)]] * y[keep]).sum()) / den
    f = max(1, int(fade_s * sr))
    w = np.ones(n)  # 1 = original, 0 = generated
    w[a:b] = 0.0
    w[len(x):] = 0.0
    if a > 0:
        lo = max(0, a - f)
        w[lo:a] = np.cos(np.linspace(0, np.pi / 2, a - lo)) ** 2
    if b < min(n, len(x)):
        hi = min(len(x), b + f)
        w[b:hi] = np.sin(np.linspace(0, np.pi / 2, hi - b)) ** 2
    xx = np.zeros_like(y)
    xx[: len(x)] = x
    z = w[:, None] * xx + (1 - w[:, None]) * y
    sf.write(str(out), np.clip(z, -1.0, 1.0), sr, subtype="PCM_16")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("stable-audio-3")
    except Exception:
        return "unknown"
