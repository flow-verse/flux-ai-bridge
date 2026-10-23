"""``flux-ai serve --json``: a warm worker that keeps models loaded between jobs.

Reads newline-delimited JSON job requests on stdin and answers with protocol
v2 events on stdout (see docs/protocol.md). Jobs run one at a time in arrival
order; backend instances (and their loaded weights) persist for the life of
the process.

Request:  {"job_id": "a1", "job": "generate", "params": {...}, "accept_licences": ["ace-step-1.5"]}
Control:  {"job": "ping"}  ->  {"event": "pong", ...}
          {"job": "shutdown"}  ->  {"event": "bye", ...} and exit 0 (EOF also exits)
"""
from __future__ import annotations

import json
import sys

from flux_ai import jobs
from flux_ai.protocol import Emitter, new_job_id, protocol_stdout


def handle(line: str, out, accept: list[str]) -> bool:
    """Process one request line. Returns False when the server should stop."""
    line = line.strip()
    if not line:
        return True
    try:
        req = json.loads(line)
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")
    except ValueError as e:
        Emitter(True, "serve", None, stream=out).error(f"bad request: {e}", error_type="BadRequest")
        return True
    job = req.get("job") or req.get("cmd") or ""
    em = Emitter(True, job, str(req.get("job_id") or new_job_id()), stream=out)
    if job == "shutdown":
        em.emit("bye")
        return False
    if job == "ping":
        em.emit("pong")
        return True
    try:
        jobs.run(job, req.get("params") or {}, em, accept=list(accept) + list(req.get("accept_licences") or []))
    except Exception as e:
        em.error(str(e), error_type=type(e).__name__)
    return True


def main(json_mode: bool = True, accept: list[str] | None = None, stdin=None) -> int:
    # serve always speaks the JSON protocol; --json is accepted for symmetry with other commands.
    stdin = stdin or sys.stdin
    with protocol_stdout(True) as out:
        Emitter(True, "serve", new_job_id(), stream=out).emit("ready", jobs=list(jobs.JOB_NAMES))
        for line in stdin:
            if not handle(line, out, accept or []):
                break
    return 0
