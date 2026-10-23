"""Magenta RealTime 2 adapter (Apache-2.0 code, CC-BY-4.0 weights), MLX path.

Install: ``pip install -e "python[magenta-rt]"`` (Python >= 3.11, Apple Silicon).

MRT2 streams 40 ms frames (25 Hz) at 48 kHz stereo and has no tempo or key
input. It does take per-frame MIDI-note and drums conditioning, so when
``bpm`` or ``key`` is given this adapter builds a *note guide*: a 4-bar chord
loop in the key (i-VI-III-VII minor, I-V-vi-IV major) with an onset on every
beat at the requested tempo, plus drums on. Pitches outside the guide stay
masked so the model can add its own notes (``extra["note_guide_strict"]``
turns them off instead). ``extra["note_guide"]=false`` disables the guide.

``extra`` keys: ``model`` ("base" = mrt2_base 2.4B default, "small" =
mrt2_small 230M), ``temperature`` (1.3), ``top_k`` (40), ``cfg_musiccoca``
(3.0), ``cfg_notes`` (1.0), ``drums`` (true), ``note_guide``,
``note_guide_strict``.

Weights download on first use from ``google/magenta-realtime-2`` into
``$FLUX_AI_CACHE/models/magenta`` (override with ``MAGENTA_HOME``).
"""
from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
from typing import Callable

import numpy as np

from flux_ai import licences, music, provenance
from flux_ai.backends import GenerateRequest, GenerateResult, check_request, finalize, report

TIERS = {"base": "mrt2_base", "small": "mrt2_small"}
HF_REPO = "google/magenta-realtime-2"
FRAME_RATE = 25
SR = 48000
# magenta-rt 2.0.3's uv.lock pins MLX 0.31.2; MLX 0.32.x fails to import the published
# .mlxfn files with "[import_function] Invalid string size." (checked 2026-10-23).
MLX_EXPORT_VERSION = "0.31.2"


def magenta_base() -> Path:
    return Path(os.environ.get("MAGENTA_HOME") or licences.cache_dir() / "models" / "magenta")


def note_guide(frames: int, bpm: float, key: str, strict: bool = False) -> list[list[int]]:
    """Per-frame 128-slot note conditioning (-1 masked, 0 off, 1 held, 2 onset)."""
    pc, mode = music.parse_key(key)
    chords = music.chord_progression(pc, mode)
    beat_frames = 60.0 / bpm * FRAME_RATE
    base = 0 if strict else -1
    rows = []
    beat = 0
    for f in range(frames):
        while int(round((beat + 1) * beat_frames)) <= f:
            beat += 1
        pos = f - int(round(beat * beat_frames))
        chord = chords[(beat // 4) % 4]
        row = [base] * 128
        held = pos < 0.8 * beat_frames
        state = 2 if pos == 0 else (1 if held else 0)
        pitches = [36 + chord[0] + (12 if chord[0] < 4 else 0)] + [60 + p for p in chord]
        for p in pitches:
            row[p] = state
        rows.append(row)
    return rows


class MagentaRTBackend:
    name = "magenta-rt"
    licence_id = "magenta-rt-2"
    modes = frozenset({"generate"})
    models = list(TIERS)
    output_kind = "audio"

    def __init__(self):
        self._systems: dict[str, object] = {}

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("magenta_rt") is None:
            return False, 'magenta-rt not installed: pip install -e "python[magenta-rt]" (Python >= 3.11)'
        if importlib.util.find_spec("mlx") is None:
            return False, "MLX not installed (needs Apple Silicon); the JAX path is not wired into this adapter"
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            return False, "the MLX path needs an Apple Silicon Mac"
        from importlib.metadata import version

        if version("mlx") != MLX_EXPORT_VERSION:
            return False, (f"MLX {version('mlx')} cannot load the published MRT2 .mlxfn files; "
                           f"install mlx=={MLX_EXPORT_VERSION} (the version magenta-rt 2.0.3 exports with)")
        return True, f"installed; weights in {magenta_base()} (downloaded on first use)"

    def _ensure_weights(self, size: str, progress) -> None:
        home = magenta_base() / "magenta-rt-v2"
        need = [home / "models" / size / f"{size}.mlxfn", home / "resources" / "musiccoca" / "spm.model"]
        if all(p.exists() for p in need):
            return
        from huggingface_hub import snapshot_download

        report(progress, 0.01, f"downloading {HF_REPO} ({size} + resources) to {home}")
        snapshot_download(
            HF_REPO, local_dir=str(home),
            allow_patterns=["resources/musiccoca/*", "resources/spectrostream/*", f"models/{size}/*"],
        )

    def _system(self, size: str, progress):
        if size not in self._systems:
            os.environ.setdefault("MAGENTA_HOME", str(magenta_base()))
            self._ensure_weights(size, progress)
            from magenta_rt import paths

            paths.set_magenta_home(magenta_base() / "magenta-rt-v2")
            from magenta_rt import MagentaRT2StdMlxfn

            report(progress, 0.04, f"loading {size} (MLX)")
            try:
                self._systems[size] = MagentaRT2StdMlxfn(size=size)
            except RuntimeError as e:
                if "import_function" in str(e):
                    import mlx.core as mx

                    raise RuntimeError(
                        f"{e} The published .mlxfn files were exported with MLX {MLX_EXPORT_VERSION}; "
                        f"this environment has MLX {mx.__version__}. Install mlx=={MLX_EXPORT_VERSION} "
                        "(newer MLX cannot read them)."
                    ) from e
                raise
        return self._systems[size]

    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult:
        check_request(self, req)
        licences.require(self.licence_id)
        tier = str(req.extra.get("model", "base"))
        if tier not in TIERS:
            raise ValueError(f"unknown magenta-rt model '{tier}'; choose from {', '.join(TIERS)}")
        size = TIERS[tier]
        started = provenance.now()
        mrt = self._system(size, progress)

        import mlx.core as mx
        from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

        mx.random.seed(int(req.seed))
        np.random.seed(int(req.seed) % (2**32))
        temperature = float(req.extra.get("temperature", 1.3))
        top_k = int(req.extra.get("top_k", 40))
        cfg = {"musiccoca": float(req.extra.get("cfg_musiccoca", 3.0)), "notes": float(req.extra.get("cfg_notes", 1.0)), "drums": 1.0}

        report(progress, 0.06, "embedding style prompt")
        emb = mrt.embed_style(req.prompt, use_mapper=True, seed=int(req.seed))
        style_len = next(c.rvq_truncation_level for c in mrt.input_configs if c.key == MUSICCOCA.key)
        style = list(mrt.tokenize_style(emb).tolist())[:style_len]
        style += [-1] * (style_len - len(style))

        frames = int(round(req.seconds * FRAME_RATE))
        use_guide = bool(req.extra.get("note_guide", req.bpm is not None or req.key is not None))
        drums = [1] if req.extra.get("drums", True) else [0]
        guide = None
        if use_guide:
            guide = note_guide(frames, float(req.bpm or 120.0), req.key or "C major", bool(req.extra.get("note_guide_strict", False)))

        # Group consecutive frames with identical conditioning into one call; keep the stream state.
        chunks, state, f = [], None, 0
        while f < frames:
            g = f + 1
            if guide is None:
                g = min(frames, f + FRAME_RATE * 2)
            else:
                while g < frames and guide[g] == guide[f]:
                    g += 1
            cond = {MUSICCOCA.key: style, DRUM_PIANOROLL.key: drums}
            if guide is not None:
                cond[PIANOROLL_WITH_ONSETS.key] = guide[f]
            wav, state = mrt.generate(conditioning=cond, cfg_scales=cfg, temperature=temperature, top_k=top_k, frames=g - f, state=state)
            chunks.append(wav.samples)
            f = g
            report(progress, 0.08 + 0.88 * f / frames, f"{f}/{frames} frames")
        audio = np.concatenate(chunks, axis=0)
        music.write_wav(req.out, audio, SR)
        report(progress, 1.0, "done")
        return finalize(
            self, req, output=req.out, model=size, model_version=f"magenta-rt {_version()} (Hugging Face {HF_REPO})",
            sample_rate=SR, seconds=audio.shape[0] / SR, started_at=started,
            backend_params={"temperature": temperature, "top_k": top_k, "cfg_scales": cfg, "note_guide": use_guide,
                            "note_guide_strict": bool(req.extra.get("note_guide_strict", False)), "drums": drums[0],
                            "frames": frames, "path": "mlxfn"},
        )


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("magenta-rt")
    except Exception:
        return "unknown"
