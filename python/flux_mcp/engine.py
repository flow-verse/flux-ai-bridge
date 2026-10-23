# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervised ``flux-engine serve`` child speaking JSON-RPC 2.0 over stdio.

The client is synchronous and thread-safe: one reader thread matches responses
to requests by id, so it works the same from the MCP server (via
``anyio.to_thread``), from the scripted eval runner and from tests, without
being tied to one event loop.

Supervision:
- the engine is started lazily on the first call;
- if it exits (crash) or a call times out, the call raises :class:`EngineCrashed`
  with the tail of the engine's stderr; the next call starts a fresh engine and,
  when the previous project had been saved to disk, reopens it (unsaved edits and
  the undo history are lost, and the error says so).
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

# JSON-RPC error codes from flux/engine/API.md
NO_PROJECT = -32000
UNKNOWN_ID = -32001
WRONG_TYPE = -32002
OP_FAILED = -32003
STATE_CONFLICT = -32004
INVALID_PARAMS = -32602


class EngineError(Exception):
    """A JSON-RPC error returned by the engine."""

    def __init__(self, code: int, message: str, method: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.method = method

    def __str__(self) -> str:
        return f"{self.message} (engine error {self.code} in {self.method})" if self.method else self.message


class EngineCrashed(Exception):
    """The engine process died or stopped answering; it will be restarted on the next call."""


class EngineNotFound(Exception):
    pass


def find_engine(explicit: str | None = None) -> str:
    """Resolve the engine binary: explicit path, then $FLUX_ENGINE, then PATH."""
    for cand in (explicit, os.environ.get("FLUX_ENGINE")):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
            raise EngineNotFound(f"flux-engine not found or not executable at '{cand}'")
    found = shutil.which("flux-engine")
    if found:
        return found
    raise EngineNotFound("flux-engine not found: pass --engine PATH, set FLUX_ENGINE, or put flux-engine on PATH")


class EngineClient:
    DEFAULT_TIMEOUT = 60.0

    def __init__(self, engine: str | None = None, extra_args: list[str] | None = None,
                 env: dict[str, str] | None = None, start_timeout: float = 30.0):
        self._engine_arg = engine
        self.extra_args = list(extra_args or [])
        self.env = env
        self.start_timeout = start_timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()          # one request in flight (the engine is sequential anyway)
        self._pending: dict[int, dict] = {}
        self._cv = threading.Condition()
        self._next_id = 0
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self._ready: dict | None = None
        self._dead_reason: str | None = None
        self.last_saved_dir: str | None = None  # reopened after a restart
        self.restarts = 0
        self._started_once = False
        self.notice: str | None = None          # one-shot message after an automatic restart

    # -- process management -------------------------------------------------
    @property
    def path(self) -> str:
        return find_engine(self._engine_arg)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def sample_rate(self) -> int | None:
        return (self._ready or {}).get("sample_rate")

    def _start(self) -> None:
        exe = self.path
        env = dict(os.environ if self.env is None else self.env)
        self._dead_reason = None
        self._stderr_tail.clear()
        self._proc = subprocess.Popen(
            [exe, "serve", *self.extra_args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, encoding="utf-8", bufsize=1,
        )
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True, name="flux-engine-stdout").start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True, name="flux-engine-stderr").start()
        with self._cv:
            ok = self._cv.wait_for(lambda: self._ready is not None or self._dead_reason is not None, self.start_timeout)
        if not ok or self._ready is None:
            reason = self._dead_reason or f"no ready message within {self.start_timeout:.0f} s"
            self._kill()
            raise EngineCrashed(f"flux-engine failed to start: {reason}{self._tail()}")

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # never let a stray line break the stream
            if not isinstance(msg, dict):
                continue
            with self._cv:
                if msg.get("method") == "ready":
                    self._ready = msg.get("params") or {}
                elif "id" in msg:
                    self._pending[msg["id"]] = msg
                self._cv.notify_all()
        with self._cv:
            if proc is self._proc:
                code = proc.wait()
                self._dead_reason = f"flux-engine exited with status {code}"
                self._ready = None
            self._cv.notify_all()

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())

    def _tail(self) -> str:
        lines = [l for l in self._stderr_tail if l.strip()][-8:]
        return ("\nengine stderr (last lines):\n" + "\n".join(lines)) if lines else ""

    def _kill(self) -> None:
        p, self._proc = self._proc, None
        self._ready = None
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def ensure_started(self) -> None:
        if self.running and self._ready is not None:
            return
        restarting = self._started_once
        self._kill()
        self._start()
        self._started_once = True
        if restarting:
            self.restarts += 1
            reopened = ""
            if self.last_saved_dir and Path(self.last_saved_dir).is_dir():
                try:
                    self._call_locked("open_project", {"dir": self.last_saved_dir}, 120.0)
                    reopened = f" Reopened the last saved project {self.last_saved_dir}; unsaved edits and undo history were lost."
                except Exception as e:  # pragma: no cover - best effort
                    reopened = f" Could not reopen {self.last_saved_dir}: {e}"
            self.notice = "flux-engine was restarted after a crash or timeout." + reopened

    def close(self) -> None:
        with self._lock:
            if self.running:
                try:
                    self._call_locked("shutdown", {}, 5.0)
                except Exception:
                    pass
                try:
                    assert self._proc is not None
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
            self._kill()

    # -- calls ----------------------------------------------------------------
    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> Any:
        """Call one method. Raises EngineError (JSON-RPC error) or EngineCrashed."""
        with self._lock:
            self.ensure_started()
            return self._call_locked(method, params or {}, timeout or self.DEFAULT_TIMEOUT)

    def _call_locked(self, method: str, params: dict, timeout: float) -> Any:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise EngineCrashed("flux-engine is not running")
        self._next_id += 1
        rid = self._next_id
        line = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            reason = self._dead_reason or "broken pipe"
            self._kill()
            raise EngineCrashed(f"flux-engine died ({reason}) during '{method}'; it will be restarted on the next call.{self._tail()}")
        with self._cv:
            ok = self._cv.wait_for(lambda: rid in self._pending or self._dead_reason is not None, timeout)
            msg = self._pending.pop(rid, None)
        if msg is None:
            if not ok:
                self._kill()
                raise EngineCrashed(
                    f"flux-engine did not answer '{method}' within {timeout:.0f} s; it was stopped and will be "
                    f"restarted on the next call.{self._tail()}")
            reason = self._dead_reason or "unknown"
            self._kill()
            raise EngineCrashed(f"flux-engine crashed ({reason}) during '{method}'; it will be restarted on the next "
                                f"call.{self._tail()}")
        if "error" in msg:
            err = msg["error"] or {}
            raise EngineError(int(err.get("code", -32603)), str(err.get("message", "unknown error")), method)
        result = msg.get("result")
        if method in ("create_project", "open_project", "save_project") and isinstance(result, dict) and result.get("dir"):
            self.last_saved_dir = result["dir"]
        return result


def log(msg: str) -> None:
    """Diagnostics go to stderr; stdout belongs to the MCP protocol."""
    print(f"flux-mcp: {msg}", file=sys.stderr, flush=True)
