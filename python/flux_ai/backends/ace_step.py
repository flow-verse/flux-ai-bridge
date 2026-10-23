"""ACE-Step 1.5 adapter (MIT), via its in-process Python API.

Install: ``pip install -e "python[ace-step]"`` (Python 3.11/3.12). On Apple
Silicon ACE-Step runs the DiT and VAE through MLX and the PyTorch parts on
MPS; the 5Hz LM planner runs through MLX.

Model tiers (``extra["model"]``):
  turbo (default, 2B, 8 steps), xl-turbo (4B), base, sft, xl-base, xl-sft.
Other ``extra`` keys: ``lm`` ("1.7B" default, "0.6B", "4B" or "none"),
``thinking`` (bool, LM chain-of-thought; default on when an LM is loaded),
``steps``, ``shift``, ``guidance_scale``, ``time_signature``,
``start``/``end`` (repaint/infill region in seconds).

Modes: generate (text2music), continue (outpaint after the context audio; the
output contains context + continuation), repaint and infill (regenerate
``start``..``end`` of the context).

Weights download on first use from Hugging Face (``ACE-Step/Ace-Step1.5`` and
sub-model repos) into ``$FLUX_AI_CACHE/models/ace-step`` (override with
``ACESTEP_CHECKPOINTS_DIR``).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Callable

from flux_ai import licences, provenance
from flux_ai.backends import GenerateRequest, GenerateResult, check_request, finalize, report

TIERS = {
    "turbo": "acestep-v15-turbo",
    "xl-turbo": "acestep-v15-xl-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
    "xl-base": "acestep-v15-xl-base",
    "xl-sft": "acestep-v15-xl-sft",
}
LMS = {"0.6B": "acestep-5Hz-lm-0.6B", "1.7B": "acestep-5Hz-lm-1.7B", "4B": "acestep-5Hz-lm-4B"}


def checkpoints_dir() -> Path:
    env = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if env:
        return Path(env).expanduser()
    return licences.cache_dir() / "models" / "ace-step"


class AceStepBackend:
    name = "ace-step"
    licence_id = "ace-step-1.5"
    modes = frozenset({"generate", "continue", "repaint", "infill"})
    models = list(TIERS)
    output_kind = "audio"

    def __init__(self):
        self._dit = None
        self._dit_cfg: str | None = None
        self._lm = None
        self._lm_name: str | None = None

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("acestep") is None:
            return False, 'acestep not installed: pip install -e "python[ace-step]" (Python 3.11/3.12)'
        if not licences.is_accepted(self.licence_id):
            return True, f"installed; weights download on first use after --accept-licence {self.licence_id}"
        return True, f"installed; checkpoints in {checkpoints_dir()}"

    # -- loading (kept warm between jobs) -------------------------------------------------
    def _load(self, tier: str, lm: str, progress) -> None:
        os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", str(checkpoints_dir()))
        ckpt = Path(os.environ["ACESTEP_CHECKPOINTS_DIR"])
        ckpt.mkdir(parents=True, exist_ok=True)
        cfg = TIERS[tier]
        if self._dit is None or self._dit_cfg != cfg:
            from acestep import model_downloader as md
            from acestep.handler import AceStepHandler

            report(progress, 0.01, f"ensuring ACE-Step weights ({cfg}) in {ckpt}")
            md.ensure_main_model(checkpoints_dir=ckpt, prefer_source="huggingface")
            if cfg != "acestep-v15-turbo":
                md.ensure_dit_model(cfg, checkpoints_dir=ckpt, prefer_source="huggingface")
            report(progress, 0.03, f"loading DiT {cfg}")
            dit = AceStepHandler()
            msg, ok = dit.initialize_service(
                project_root=str(ckpt.parent), config_path=cfg, device="auto",
                offload_to_cpu=False, prefer_source="huggingface",
            )
            if not ok:
                raise RuntimeError(f"ACE-Step DiT init failed: {msg}")
            self._dit, self._dit_cfg = dit, cfg
        want_lm = None if lm == "none" else LMS[lm]
        if want_lm != self._lm_name:
            from acestep.llm_inference import LLMHandler

            if want_lm is None:
                self._lm, self._lm_name = LLMHandler(), None  # uninitialised handler = LM disabled
            else:
                from acestep import model_downloader as md

                if want_lm != md.DEFAULT_LM_MODEL:
                    md.ensure_lm_model(want_lm, checkpoints_dir=ckpt, prefer_source="huggingface")
                report(progress, 0.06, f"loading LM {want_lm}")
                handler = LLMHandler()
                backend = "mlx" if _is_apple_silicon() else "pt"
                msg, ok = handler.initialize(checkpoint_dir=str(ckpt), lm_model_path=want_lm, backend=backend, device="auto")
                if not ok:
                    raise RuntimeError(f"ACE-Step LM init failed: {msg}")
                self._lm, self._lm_name = handler, want_lm
        elif self._lm is None:
            from acestep.llm_inference import LLMHandler

            self._lm = LLMHandler()

    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult:
        check_request(self, req)
        licences.require(self.licence_id)
        tier = str(req.extra.get("model", "turbo"))
        if tier not in TIERS:
            raise ValueError(f"unknown ace-step model '{tier}'; choose from {', '.join(TIERS)}")
        lm = str(req.extra.get("lm", "1.7B"))
        if lm != "none" and lm not in LMS:
            raise ValueError(f"unknown ace-step lm '{lm}'; choose from none, {', '.join(LMS)}")
        started = provenance.now()
        self._load(tier, lm, progress)

        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        is_turbo = "turbo" in tier
        thinking = bool(req.extra.get("thinking", lm != "none"))
        kw = dict(
            task_type="text2music",
            caption=req.prompt,
            lyrics="[Instrumental]" if req.instrumental and not req.lyrics else (req.lyrics or ""),
            instrumental=req.instrumental,
            bpm=int(round(req.bpm)) if req.bpm else None,
            keyscale=_ace_key(req.key),
            timesignature=str(req.extra.get("time_signature", "4")),
            duration=float(req.seconds),
            inference_steps=int(req.extra.get("steps", 8 if is_turbo else 50)),
            shift=float(req.extra.get("shift", 3.0 if is_turbo else 1.0)),
            guidance_scale=float(req.extra.get("guidance_scale", 1.0 if is_turbo else 7.0)),
            seed=int(req.seed),
            thinking=thinking and lm != "none",
            use_cot_caption=bool(req.extra.get("cot_caption", False)),
        )
        if req.mode != "generate":
            from flux_ai.music import read_wav

            ctx, sr = read_wav(req.context_audio)
            ctx_s = ctx.shape[0] / sr
            if req.mode == "continue":
                start, end = ctx_s, ctx_s + float(req.seconds)
            else:
                start = float(req.extra.get("start", 0.0))
                end = float(req.extra.get("end", min(ctx_s, start + req.seconds)))
            kw.update(task_type="repaint", src_audio=str(req.context_audio), repainting_start=start,
                      repainting_end=end, chunk_mask_mode="explicit")
            kw.pop("duration")
        params = GenerationParams(**kw)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(req.seed)], audio_format="wav")

        def _progress(value, desc=None, *a, **k):
            report(progress, 0.08 + 0.87 * float(value), str(desc or ""))

        with tempfile.TemporaryDirectory(prefix="flux-ace-") as tmp:
            res = generate_music(self._dit, self._lm, params, config, save_dir=tmp, progress=_progress)
            if not res.success:
                raise RuntimeError(f"ACE-Step generation failed: {res.error or res.status_message}")
            audio = res.audios[0]
            tensor = audio["tensor"]  # [channels, samples], float32 CPU
            sr = int(audio.get("sample_rate", 48000))
            from flux_ai.music import write_wav

            x = tensor.numpy().T if hasattr(tensor, "numpy") else tensor.T
            write_wav(req.out, x, sr)
        report(progress, 1.0, "done")
        p = audio.get("params", {}) or {}
        lm_meta = {k: p.get(k) for k in ("cot_bpm", "cot_keyscale", "cot_caption", "cot_timesignature") if p.get(k)}
        return finalize(
            self, req, output=req.out, model=f"{TIERS[tier]}" + (f"+{LMS[lm]}" if lm != "none" else ""),
            model_version=f"ace-step {_version()} (Hugging Face ACE-Step/Ace-Step1.5)",
            sample_rate=sr, seconds=x.shape[0] / sr, started_at=started,
            backend_params={**{k: v for k, v in kw.items() if k not in ("caption", "lyrics")}, "lm": lm,
                            "lm_metadata": lm_meta, "time_costs": res.extra_outputs.get("time_costs", {})},
        )


def _ace_key(key: str | None) -> str:
    if not key:
        return ""
    from flux_ai.music import normalise_key

    return normalise_key(key) or ""


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("ace-step")
    except Exception:
        return "unknown"


def _is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"
