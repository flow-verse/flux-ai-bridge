"""flux-ai job protocol v2 (JSON lines on stdout).

Every event is one JSON object per line. v2 adds two fields to every event,
``protocol_version`` (always 2) and ``job_id``, and a new ``progress`` event.
The v1 events keep their v1 fields, so v1 parsers (``worker.rs``) still work:

    {"event": "started",  "protocol_version": 2, "job_id": "…", "job": "generate", "input": "…"}
    {"event": "progress", "protocol_version": 2, "job_id": "…", "job": "generate", "fraction": 0.4, "message": "…"}
    {"event": "done",     "protocol_version": 2, "job_id": "…", "job": "generate", "outputs": ["…"]}
    {"event": "error",    "protocol_version": 2, "job_id": "…", "job": "generate", "message": "…"}

See docs/protocol.md for the full description, including ``flux-ai serve``.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import uuid
from typing import IO, Iterator

PROTOCOL_VERSION = 2


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


class Emitter:
    """Writes protocol events for one job. Disabled emitters are no-ops."""

    _lock = threading.Lock()  # serve mode may emit from more than one thread

    def __init__(self, enabled: bool, job: str, job_id: str | None = None, stream: IO[str] | None = None):
        self.enabled = enabled
        self.job = job
        self.job_id = job_id or new_job_id()
        self._stream = stream

    def emit(self, event: str, **fields) -> None:
        if not self.enabled:
            return
        rec = {"event": event, "protocol_version": PROTOCOL_VERSION, "job_id": self.job_id, "job": self.job}
        rec.update(fields)
        line = json.dumps(rec, default=str)
        stream = self._stream or sys.stdout
        with self._lock:
            stream.write(line + "\n")
            stream.flush()

    def started(self, input: str, **extra) -> None:
        self.emit("started", input=input, **extra)

    def progress(self, fraction: float, message: str = "") -> None:
        """Emit progress, throttled to steps of at least 1% (first and final updates always go out)."""
        fraction = max(0.0, min(1.0, float(fraction)))
        last = getattr(self, "_last_progress", None)
        if last is not None and fraction < 1.0 and fraction - last < 0.01:
            return
        self._last_progress = fraction
        self.emit("progress", fraction=round(fraction, 4), message=message)

    def done(self, outputs: list, **extra) -> None:
        self.emit("done", outputs=[str(o) for o in outputs], **extra)

    def error(self, message: str, **extra) -> None:
        self.emit("error", message=message, **extra)


@contextlib.contextmanager
def protocol_stdout(enabled: bool) -> Iterator[IO[str]]:
    """Reserve stdout for protocol events while the body runs.

    Model libraries print freely (progress bars, logging, C-level writes from
    MLX/TFLite). In JSON mode everything except protocol events is moved to
    stderr: at the file-descriptor level when stdout is a real fd, otherwise
    (e.g. under pytest's capsys) at the Python level only.
    """
    if not enabled:
        yield sys.stdout
        return
    real = sys.stdout
    try:
        fd = real.fileno()
    except (AttributeError, OSError, ValueError):
        fd = None
    if fd is None:
        with contextlib.redirect_stdout(sys.stderr):
            yield real
        return
    real.flush()
    saved = os.dup(fd)
    proto = os.fdopen(os.dup(saved), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), fd)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield proto
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        proto.flush()
        os.dup2(saved, fd)
        os.close(saved)
        proto.close()
