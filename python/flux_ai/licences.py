"""Model weight licences and the once-per-model acceptance record.

Weights are never bundled. Before a backend downloads or loads weights, the
user (or an agent, non-interactively via ``--accept-licence ID``) accepts that
model's licence once; the acceptance is stored in
``$FLUX_AI_CACHE/licences.json`` (default ``~/.cache/flux-ai/licences.json``).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Licence:
    id: str
    model: str
    name: str
    spdx: str
    url: str
    summary: str
    commercial_use: str
    code_licence: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


LICENCES: dict[str, Licence] = {
    "ace-step-1.5": Licence(
        id="ace-step-1.5",
        model="ACE-Step 1.5 (DiT 2B/4B XL, 5Hz LM, VAE, Qwen3 text encoder)",
        name="MIT",
        spdx="MIT",
        url="https://huggingface.co/ACE-Step/Ace-Step1.5",
        code_licence="MIT (https://github.com/ace-step/ACE-Step-1.5)",
        summary="Permissive; keep the copyright notice.",
        commercial_use="allowed",
        notes=("Training data is not documented by upstream.",),
    ),
    "magenta-rt-2": Licence(
        id="magenta-rt-2",
        model="Magenta RealTime 2 (mrt2_base 2.4B / mrt2_small 230M, MusicCoCa, SpectroStream)",
        name="CC-BY-4.0 weights, Apache-2.0 code, plus the model card's use terms",
        spdx="CC-BY-4.0",
        url="https://huggingface.co/google/magenta-realtime-2",
        code_licence="Apache-2.0 (https://github.com/magenta/magenta-realtime)",
        summary="Attribution required; the model card adds terms against infringing use. "
        "Google claims no rights in outputs.",
        commercial_use="allowed with attribution",
        notes=("Trained on ~71k hours of mostly instrumental stock music; licence status of that data not stated.",),
    ),
    "stable-audio-3": Licence(
        id="stable-audio-3",
        model="Stable Audio 3 Small-Music / Medium",
        name="Stability AI Community License",
        spdx="LicenseRef-Stability-AI-Community",
        url="https://huggingface.co/stabilityai/stable-audio-3-medium",
        code_licence="MIT (https://github.com/Stability-AI/stable-audio-3)",
        summary="Free for individuals and organisations under USD 1M annual revenue; "
        "above that an Enterprise licence from Stability AI is required. Outputs are yours.",
        commercial_use="capped: USD 1M annual revenue",
        notes=(
            "Opt-in only. The weights are also gated on Hugging Face: accept the terms there and log in (hf auth login).",
        ),
    ),
    "amt": Licence(
        id="amt",
        model="Anticipatory Music Transformer (stanford-crfm/music-{small,medium,large}-800k)",
        name="Apache-2.0",
        spdx="Apache-2.0",
        url="https://huggingface.co/stanford-crfm/music-medium-800k",
        code_licence="Apache-2.0 (https://github.com/jthickstun/anticipation)",
        summary="Permissive; keep the licence and NOTICE.",
        commercial_use="allowed",
        notes=("Trained on the Lakh MIDI dataset.",),
    ),
}


class LicenceNotAccepted(RuntimeError):
    def __init__(self, licence_id: str):
        lic = LICENCES.get(licence_id)
        desc = f"{lic.model}: {lic.name} ({lic.url}). {lic.summary}" if lic else licence_id
        super().__init__(
            f"licence '{licence_id}' has not been accepted. {desc} "
            f"Review it, then re-run with --accept-licence {licence_id} (recorded once in {licence_file()})."
        )
        self.licence_id = licence_id


def cache_dir() -> Path:
    return Path(os.environ.get("FLUX_AI_CACHE") or Path.home() / ".cache" / "flux-ai")


def licence_file() -> Path:
    return cache_dir() / "licences.json"


def _load() -> dict:
    try:
        return json.loads(licence_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def accepted(licence_id: str) -> dict | None:
    return _load().get(licence_id)


def is_accepted(licence_id: str) -> bool:
    if licence_id not in LICENCES:  # built-in / no weights: nothing to accept
        return True
    return accepted(licence_id) is not None


def accept(licence_id: str, via: str = "cli") -> dict:
    if licence_id not in LICENCES:
        raise KeyError(f"unknown licence id '{licence_id}'; known: {', '.join(sorted(LICENCES))}")
    from flux_ai import __version__

    data = _load()
    if licence_id in data:
        return data[licence_id]
    lic = LICENCES[licence_id]
    rec = {
        "licence_id": licence_id,
        "licence": lic.name,
        "spdx": lic.spdx,
        "url": lic.url,
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "via": via,
        "flux_ai_version": __version__,
    }
    data[licence_id] = rec
    path = licence_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return rec


def require(licence_id: str) -> None:
    """Raise LicenceNotAccepted unless the licence was accepted before."""
    if not is_accepted(licence_id):
        raise LicenceNotAccepted(licence_id)


def prompt_accept(licence_id: str) -> bool:
    """Interactive acceptance on a terminal; False when not interactive or declined."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return False
    lic = LICENCES[licence_id]
    print(f"\n{lic.model}\n  weights licence: {lic.name}\n  {lic.url}\n  {lic.summary}", file=sys.stderr)
    for n in lic.notes:
        print(f"  note: {n}", file=sys.stderr)
    print("Accept this licence? [y/N] ", end="", file=sys.stderr, flush=True)
    if sys.stdin.readline().strip().lower() in ("y", "yes"):
        accept(licence_id, via="interactive")
        return True
    return False


def describe(licence_id: str) -> dict:
    lic = LICENCES.get(licence_id)
    return asdict(lic) if lic else {"id": licence_id}
