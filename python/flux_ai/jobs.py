"""Job execution shared by the one-shot CLI and the ``serve`` warm worker."""
from __future__ import annotations

import time
from pathlib import Path

from flux_ai import licences
from flux_ai.protocol import Emitter

JOB_NAMES = ("separate", "transcribe", "generate", "backends")


def _accept(ids: list[str] | None) -> None:
    for lid in ids or []:
        licences.accept(lid, via="--accept-licence")


def run(job: str, params: dict, em: Emitter, accept: list[str] | None = None, interactive: bool = False):
    """Run one job, emitting started/progress/done. Raises on failure (caller emits error)."""
    if job not in JOB_NAMES:
        raise ValueError(f"unknown job '{job}'; expected one of {', '.join(JOB_NAMES)}")
    if job in ("separate", "transcribe"):
        from flux_ai import cli

        src, out = Path(params["input"]), Path(params["out"])
        em.started(input=str(src))
        outputs = cli.JOBS[job](src, out, quiet=em.enabled)
        em.done(outputs)
        return outputs
    if job == "backends":
        from flux_ai.backends import list_backends

        _accept(accept)
        em.started(input="")
        bl = list_backends()
        em.done([], backends=bl)
        return bl
    return _generate(params, em, accept, interactive)


def _generate(params: dict, em: Emitter, accept: list[str] | None, interactive: bool):
    from flux_ai.backends import GenerateRequest, get_backend

    required = ("prompt", "out")
    missing = [k for k in required if params.get(k) in (None, "")]
    if missing:
        raise ValueError(f"generate: missing {', '.join(missing)}")
    em.started(input=str(params["prompt"]), backend=params.get("backend", "ace-step"), out=str(params["out"]))
    _accept(accept)
    backend = get_backend(params.get("backend", "ace-step"))
    if not licences.is_accepted(backend.licence_id):
        if not (interactive and licences.prompt_accept(backend.licence_id)):
            raise licences.LicenceNotAccepted(backend.licence_id)
    ok, reason = backend.available()
    if not ok:
        raise RuntimeError(f"backend '{backend.name}' is not available: {reason}")
    req = GenerateRequest(
        prompt=params["prompt"],
        seconds=float(params.get("seconds", 30.0)),
        out=Path(params["out"]),
        seed=int(params.get("seed", 0)),
        bpm=float(params["bpm"]) if params.get("bpm") is not None else None,
        key=params.get("key"),
        mode=params.get("mode", "generate"),
        context_audio=Path(params["context"]) if params.get("context") else None,
        lyrics=params.get("lyrics"),
        instrumental=bool(params.get("instrumental", True)),
        extra=dict(params.get("extra") or {}),
    )
    t0 = time.perf_counter()
    res = backend.generate(req, progress=em.progress)
    elapsed = time.perf_counter() - t0
    sidecar = res.provenance.get("sidecar")
    em.done(
        [res.audio] + ([sidecar] if sidecar else []),
        backend=res.backend,
        model=res.model,
        sample_rate=res.sample_rate,
        seconds=round(res.seconds, 3),
        elapsed_s=round(elapsed, 3),
        provenance=sidecar,
    )
    return res


def print_human(job: str, result) -> None:
    if job == "backends":
        for b in result:
            flag = "ok " if b["available"] else "-- "
            tags = ("default " if b["default"] else "") + ("opt-in " if b["opt_in"] else "")
            acc = "accepted" if b["licence_accepted"] else f"--accept-licence {b['licence_id']}"
            print(f"{flag}{b['name']:<15} {tags:<9}{b['licence']}  [{acc}]")
            print(f"     {b['description']}")
            if not b["available"]:
                print(f"     unavailable: {b['reason']}")
    elif job == "generate":
        print(f"{result.audio}  ({result.seconds:.1f} s @ {result.sample_rate} Hz, {result.backend}/{result.model})")
        print(f"provenance: {result.provenance.get('sidecar')}")
