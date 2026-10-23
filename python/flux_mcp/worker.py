# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``flux-ai serve`` warm worker and one-shot ``flux-ai`` jobs (protocol v2).

``AiWorker`` keeps one ``flux-ai serve`` process alive so generation models stay
loaded between jobs. It is started lazily on the first job (most sessions never
generate) and restarted if it dies. Jobs run one at a time, as the worker does.

``run_oneshot`` runs ``flux-ai <args> --json`` for subcommands that the warm
worker does not serve (``longform``) and streams its events the same way.
"""
from __future__ import annotations

import collections
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Callable

ProgressFn = Callable[[float, str], None]


class JobError(Exception):
    def __init__(self, message: str, error_type: str = ""):
        super().__init__(message)
        self.error_type = error_type


class WorkerUnavailable(Exception):
    pass


def flux_ai_command(explicit: str | None = None) -> list[str]:
    """How to start flux-ai: explicit command, $FLUX_AI, this interpreter if flux_ai is importable, or PATH."""
    cand = explicit or os.environ.get("FLUX_AI")
    if cand:
        return cand.split()
    try:
        import flux_ai  # noqa: F401

        return [sys.executable, "-m", "flux_ai.cli"]
    except ImportError:
        pass
    found = shutil.which("flux-ai")
    if found:
        return [found]
    raise WorkerUnavailable("flux-ai not found: install flux-ai in this environment, set FLUX_AI, or pass --flux-ai")


class AiWorker:
    def __init__(self, command: list[str] | None = None, env: dict[str, str] | None = None):
        self._command = command
        self.env = env
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=40)

    @property
    def command(self) -> list[str]:
        return self._command or flux_ai_command()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> None:
        self._events = queue.Queue()
        self._proc = subprocess.Popen(
            [*self.command, "serve", "--json"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ if self.env is None else self.env), text=True, encoding="utf-8", bufsize=1,
        )
        threading.Thread(target=_pump, args=(self._proc.stdout, self._events), daemon=True).start()
        threading.Thread(target=_tail, args=(self._proc.stderr, self._stderr_tail), daemon=True).start()
        ev = self._next_event(60.0)
        if ev is None or ev.get("event") != "ready":
            self.close()
            raise WorkerUnavailable("flux-ai serve did not start" + self._tail())

    def _next_event(self, timeout: float) -> dict | None:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _tail(self) -> str:
        lines = [l for l in self._stderr_tail if l.strip()][-8:]
        return ("\nflux-ai stderr (last lines):\n" + "\n".join(lines)) if lines else ""

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        if p.poll() is None:
            try:
                assert p.stdin is not None
                p.stdin.write(json.dumps({"job": "shutdown"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def run(self, job: str, params: dict | None = None, accept_licences: list[str] | None = None,
            progress: ProgressFn | None = None, timeout: float = 3600.0) -> dict:
        """Run one job and return its ``done`` event. Raises JobError on an ``error`` event."""
        with self._lock:
            if not self.running:
                self._start()
            assert self._proc is not None and self._proc.stdin is not None
            job_id = uuid.uuid4().hex[:12]
            req = {"job_id": job_id, "job": job, "params": params or {}}
            if accept_licences:
                req["accept_licences"] = list(accept_licences)
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.close()
                raise JobError("flux-ai worker died; it will be restarted on the next job" + self._tail(), "WorkerCrashed")
            return _collect(job_id, self._next_event, lambda: self.running, progress, timeout, self.close, self._tail)


def _pump(stream, q: queue.Queue) -> None:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # protocol: ignore non-JSON lines
        if isinstance(ev, dict):
            q.put(ev)
    q.put({"event": "__eof__"})


def _tail(stream, dq) -> None:
    for line in stream:
        dq.append(line.rstrip())


def _collect(job_id, next_event, alive, progress, timeout, on_dead, tail) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            on_dead()
            raise JobError(f"job timed out after {timeout:.0f} s; the worker was stopped", "Timeout")
        ev = next_event(min(remaining, 1.0))
        if ev is None:
            if not alive():
                on_dead()
                raise JobError("flux-ai worker exited during the job" + tail(), "WorkerCrashed")
            continue
        if ev.get("event") == "__eof__":
            on_dead()
            raise JobError("flux-ai worker exited during the job" + tail(), "WorkerCrashed")
        if ev.get("job_id") not in (job_id, None):
            continue
        kind = ev.get("event")
        if kind == "progress" and progress is not None:
            progress(float(ev.get("fraction", 0.0)), str(ev.get("message", "")))
        elif kind == "done":
            return ev
        elif kind == "error":
            raise JobError(str(ev.get("message", "job failed")), str(ev.get("error_type", "")))


def run_oneshot(args: list[str], command: list[str] | None = None, progress: ProgressFn | None = None,
                timeout: float = 24 * 3600.0, env: dict[str, str] | None = None) -> dict:
    """Run ``flux-ai --json --job-id ID <args>`` and return its done event."""
    cmd = command or flux_ai_command()
    job_id = uuid.uuid4().hex[:12]
    proc = subprocess.Popen([*cmd, "--json", "--job-id", job_id, *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            env=dict(os.environ if env is None else env))
    q: queue.Queue = queue.Queue()
    dq: collections.deque[str] = collections.deque(maxlen=40)
    threading.Thread(target=_pump, args=(proc.stdout, q), daemon=True).start()
    threading.Thread(target=_tail, args=(proc.stderr, dq), daemon=True).start()

    def next_event(t):
        try:
            return q.get(timeout=t)
        except queue.Empty:
            return None

    def tail():
        lines = [l for l in dq if l.strip()][-8:]
        return ("\nflux-ai stderr (last lines):\n" + "\n".join(lines)) if lines else ""

    try:
        return _collect(job_id, next_event, lambda: proc.poll() is None or not q.empty(), progress, timeout,
                        proc.kill, tail)
    finally:
        if proc.poll() is None:
            proc.kill()


def has_subcommand(name: str, command: list[str] | None = None) -> bool:
    """True if this flux-ai has the given subcommand (checked via --help, no side effects)."""
    try:
        cmd = command or flux_ai_command()
        out = subprocess.run([*cmd, name, "--help"], capture_output=True, text=True, timeout=60)
    except (WorkerUnavailable, OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0
