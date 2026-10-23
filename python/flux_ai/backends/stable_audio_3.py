"""Stable Audio 3 adapter (opt-in; Stability AI Community License, USD 1M revenue cap).

Install in a SEPARATE venv: ``stable-audio-3`` pins torch==2.7.1 and
transformers>=5.8, which conflict with ACE-Step 1.5::

    uv venv .venv-sa3 --python 3.12
    uv pip install -p .venv-sa3 -e "python[stable-audio-3]"

The weights are gated on Hugging Face: accept the terms on the model page
(stabilityai/stable-audio-3-small-music or -medium) with your account and log
in (``hf auth login``). Flux additionally records the licence acceptance once
(``--accept-licence stable-audio-3``).

Models (``extra["model"]``): ``small-music`` (default, 433M, up to 120 s, CPU
or MPS) and ``medium`` (1.4B, up to 380 s; upstream targets CUDA with Flash
Attention 2). Modes: generate, continue (output = context + continuation),
repaint/infill (``extra["start"]``/``extra["end"]``). There is no BPM/key
input; they are appended to the prompt text ("…, 122 BPM, A minor").
"""
from __future__ import annotations

import importlib.util
from typing import Callable

from flux_ai import licences, provenance
from flux_ai.backends import GenerateRequest, GenerateResult, check_request, finalize, report

TIERS = {"small-music": "small-music", "medium": "medium"}
HF_REPOS = {"small-music": "stabilityai/stable-audio-3-small-music", "medium": "stabilityai/stable-audio-3-medium"}
MAX_SECONDS = {"small-music": 120.0, "medium": 380.0}


class StableAudio3Backend:
    name = "stable-audio-3"
    licence_id = "stable-audio-3"
    modes = frozenset({"generate", "continue", "repaint", "infill"})
    models = list(TIERS)
    output_kind = "audio"

    def __init__(self):
        self._models: dict[str, object] = {}

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("stable_audio_3") is None:
            return False, (
                "stable-audio-3 not installed. It pins torch==2.7.1 and transformers>=5.8, which conflict with "
                'ace-step, so install it in its own venv: uv pip install -p .venv-sa3 -e "python[stable-audio-3]". '
                "Weights are gated on Hugging Face: accept the terms at "
                "https://huggingface.co/stabilityai/stable-audio-3-small-music and run `hf auth login`."
            )
        if not licences.is_accepted(self.licence_id):
            return True, f"installed; opt-in: needs --accept-licence {self.licence_id} (USD 1M revenue cap) and HF gate access"
        return True, "installed; weights download on first use (HF gate access required)"

    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult:
        check_request(self, req)
        licences.require(self.licence_id)
        tier = str(req.extra.get("model", "small-music"))
        if tier not in TIERS:
            raise ValueError(f"unknown stable-audio-3 model '{tier}'; choose from {', '.join(TIERS)}")
        started = provenance.now()
        import torch
        from stable_audio_3 import StableAudioModel

        if tier not in self._models:
            report(progress, 0.02, f"loading Stable Audio 3 {tier} (gated HF repo {HF_REPOS[tier]})")
            self._models[tier] = StableAudioModel.from_pretrained(TIERS[tier])
        model = self._models[tier]
        prompt = req.prompt + "".join(
            s for s in ((f", {req.bpm:g} BPM" if req.bpm else ""), (f", {req.key}" if req.key else ""))
        )
        kw = dict(prompt=prompt, duration=float(req.seconds), seed=int(req.seed),
                  steps=int(req.extra.get("steps", 8)), cfg_scale=float(req.extra.get("cfg_scale", 1.0)))
        if req.mode != "generate":
            import torchaudio

            wav, sr_in = torchaudio.load(str(req.context_audio))
            ctx_s = wav.shape[-1] / sr_in
            if req.mode == "continue":
                start, end, total = ctx_s, ctx_s + req.seconds, ctx_s + req.seconds
            else:
                start = float(req.extra.get("start", 0.0))
                end = float(req.extra.get("end", min(ctx_s, start + req.seconds)))
                total = ctx_s
            if total > MAX_SECONDS[tier]:
                raise ValueError(f"{tier} generates at most {MAX_SECONDS[tier]:.0f} s in one call")
            kw.update(inpaint_audio=(sr_in, wav), inpaint_mask_start_seconds=start, inpaint_mask_end_seconds=end, duration=total)
        elif req.seconds > MAX_SECONDS[tier]:
            raise ValueError(f"{tier} generates at most {MAX_SECONDS[tier]:.0f} s in one call")
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
            backend_params={k: v for k, v in kw.items() if k not in ("inpaint_audio",)},
        )


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("stable-audio-3")
    except Exception:
        return "unknown"
