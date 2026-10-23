"""Generation backend interface and registry.

Every backend implements :class:`Backend`. Heavy dependencies are imported
lazily inside each adapter, so ``list_backends()`` and the base CLI work
without them. ``get_backend(name)`` returns one cached instance per name, so a
long-running worker (``flux-ai serve``) keeps models loaded between jobs.

Every backend writes the audio (or MIDI) file and a provenance sidecar
``<out>.provenance.json`` and returns both via :class:`GenerateResult`.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

MODES = frozenset({"generate", "continue", "repaint", "infill"})

ProgressFn = Callable[[float, str], None]


@dataclass
class GenerateRequest:
    prompt: str
    seconds: float
    out: Path                      # output audio file (.wav)
    seed: int = 0
    bpm: float | None = None
    key: str | None = None         # e.g. "A minor"
    mode: str = "generate"         # generate | continue | repaint | infill
    context_audio: Path | None = None
    lyrics: str | None = None
    instrumental: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class GenerateResult:
    audio: Path
    sample_rate: int
    seconds: float
    backend: str
    model: str
    provenance: dict


@runtime_checkable
class Backend(Protocol):
    name: str
    licence_id: str
    modes: frozenset[str]

    def available(self) -> tuple[bool, str]: ...   # (ok, reason) without loading weights

    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult: ...


# name -> (module, class, description, default backend?, opt-in?)
_REGISTRY: dict[str, tuple[str, str, str, bool, bool]] = {
    "ace-step": ("flux_ai.backends.ace_step", "AceStepBackend", "ACE-Step 1.5: full songs 10 s-10 min, BPM/key conditioning, repaint/extend", True, False),
    "magenta-rt": ("flux_ai.backends.magenta_rt", "MagentaRTBackend", "Magenta RealTime 2 (MLX): streaming instrumental, text + MIDI-note control", True, False),
    "stable-audio-3": ("flux_ai.backends.stable_audio_3", "StableAudio3Backend", "Stable Audio 3 Small/Medium: instrumental beds, inpainting (opt-in, $1M revenue cap)", False, True),
    "amt": ("flux_ai.backends.amt", "AMTBackend", "Anticipatory Music Transformer: symbolic MIDI generate/continue/infill/accompany (.mid)", False, True),
    "sine-test": ("flux_ai.backends.sine_test", "SineTestBackend", "Deterministic tone generator for tests and CI (no weights)", False, False),
}

_INSTANCES: dict[str, Backend] = {}


def backend_names() -> list[str]:
    return list(_REGISTRY)


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend '{name}'; known: {', '.join(_REGISTRY)}")
    if name not in _INSTANCES:
        mod, cls, *_ = _REGISTRY[name]
        _INSTANCES[name] = getattr(importlib.import_module(mod), cls)()
    return _INSTANCES[name]


def list_backends() -> list[dict]:
    from flux_ai import licences

    out = []
    for name, (_, _, desc, default, opt_in) in _REGISTRY.items():
        b = get_backend(name)
        try:
            ok, reason = b.available()
        except Exception as e:  # never let one broken adapter hide the others
            ok, reason = False, f"availability check failed: {e}"
        lic = licences.LICENCES.get(b.licence_id)
        out.append({
            "name": name,
            "description": desc,
            "default": default,
            "opt_in": opt_in,
            "available": ok,
            "reason": reason,
            "modes": sorted(b.modes),
            "models": list(getattr(b, "models", [])),
            "output": getattr(b, "output_kind", "audio"),
            "licence_id": b.licence_id,
            "licence": lic.name if lic else "none (built-in)",
            "licence_url": lic.url if lic else None,
            "commercial_use": lic.commercial_use if lic else "n/a",
            "licence_accepted": licences.is_accepted(b.licence_id),
        })
    return out


def check_request(backend: Backend, req: GenerateRequest) -> None:
    """Validation shared by all adapters."""
    if req.mode not in MODES:
        raise ValueError(f"unknown mode '{req.mode}'; expected one of {sorted(MODES)}")
    if req.mode not in backend.modes:
        raise ValueError(f"backend '{backend.name}' does not support mode '{req.mode}' (supports {sorted(backend.modes)})")
    if req.mode != "generate" and not req.context_audio:
        raise ValueError(f"mode '{req.mode}' needs context audio (--context)")
    if req.context_audio and not Path(req.context_audio).is_file():
        raise FileNotFoundError(f"context file not found: {req.context_audio}")
    if req.seconds <= 0:
        raise ValueError("seconds must be > 0")


def finalize(
    backend: Backend,
    req: GenerateRequest,
    *,
    output: Path,
    model: str,
    model_version: str,
    sample_rate: int,
    seconds: float,
    started_at: str,
    backend_params: dict | None = None,
) -> GenerateResult:
    """Write the provenance sidecar and build the result."""
    from flux_ai import provenance

    rec = provenance.build(
        req,
        backend=backend.name,
        model=model,
        model_version=model_version,
        licence_id=backend.licence_id,
        started_at=started_at,
        finished_at=provenance.now(),
        output=output,
        sample_rate=sample_rate,
        seconds=seconds,
        backend_params=backend_params,
    )
    rec["sidecar"] = str(provenance.sidecar_path(output))
    provenance.write(output, rec)
    return GenerateResult(audio=Path(output), sample_rate=sample_rate, seconds=seconds, backend=backend.name, model=model, provenance=rec)


def report(progress: ProgressFn | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(fraction, message)
