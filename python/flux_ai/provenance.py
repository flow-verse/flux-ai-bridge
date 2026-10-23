"""Per-output provenance records, written as a ``<out>.provenance.json`` sidecar."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from flux_ai import licences

if TYPE_CHECKING:
    from flux_ai.backends import GenerateRequest

SCHEMA = "flux-ai/provenance@1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sidecar_path(out: Path) -> Path:
    out = Path(out)
    return out.with_name(out.name + ".provenance.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(
    req: "GenerateRequest",
    *,
    backend: str,
    model: str,
    model_version: str,
    licence_id: str,
    started_at: str,
    finished_at: str,
    output: Path,
    sample_rate: int,
    seconds: float,
    backend_params: dict | None = None,
) -> dict:
    from flux_ai import __version__

    lic = licences.LICENCES.get(licence_id)
    return {
        "schema": SCHEMA,
        "ai_generated": True,
        "flux_ai_version": __version__,
        "backend": backend,
        "model": {
            "id": model,
            "version": model_version,
            "weights_licence": {
                "id": licence_id,
                "name": lic.name if lic else licence_id,
                "spdx": lic.spdx if lic else None,
                "url": lic.url if lic else None,
            },
            "licence_acceptance": licences.accepted(licence_id),
        },
        "prompt": req.prompt,
        "lyrics": req.lyrics,
        "seed": req.seed,
        "conditioning": {
            "mode": req.mode,
            "seconds": req.seconds,
            "bpm": req.bpm,
            "key": req.key,
            "instrumental": req.instrumental,
            "context_audio": str(req.context_audio) if req.context_audio else None,
            "context_audio_sha256": sha256(req.context_audio) if req.context_audio and Path(req.context_audio).is_file() else None,
            "extra": req.extra,
        },
        "backend_params": backend_params or {},
        "started_at": started_at,
        "finished_at": finished_at,
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "sample_rate": sample_rate,
            "seconds": round(float(seconds), 3),
        },
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]},
    }


def write(out: Path, record: dict) -> Path:
    p = sidecar_path(out)
    p.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return p
