# SPDX-License-Identifier: GPL-3.0-or-later
"""Eval runners: scripted (reference tool calls) and LLM (a model drives the tools)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client

from flux_mcp.evals.tasks import TASKS, Ctx, Task
from flux_mcp.server import Config, Session, build_server

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are a music production assistant operating the Flux DAW through its MCP tools. Complete the user's
request with the tools, then reply with one short sentence saying what you did. The project the request
refers to is already open; the instrument plugin called "Test Synth" is installed. Do not ask questions:
the request is complete as given."""


@dataclass
class TaskResult:
    id: str
    passed: bool
    reason: str = ""
    tool_calls: list[str] = field(default_factory=list)
    seconds: float = 0.0
    model: str | None = None
    turns: int = 0


async def _prepare(task: Task, client: Client, session: Session, base: Path) -> Ctx:
    tdir = base / task.id
    tdir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(client=client, session=session, dir=tdir)
    await ctx.tool("new_project", dir=str(tdir / "project"), bpm=task.bpm, discard_unsaved=True)
    if task.setup:
        await task.setup(ctx)
    ctx.calls.clear()
    return ctx


def _check(task: Task, ctx: Ctx) -> tuple[bool, str]:
    try:
        task.check(ctx)
        return True, ""
    except AssertionError as e:
        return False, str(e) or "check failed"
    except Exception as e:  # a crashing checker is a failure, with the reason
        return False, f"checker error: {type(e).__name__}: {e}"


async def run_suite(mode: str = "scripted", task_ids: list[str] | None = None, base: Path | None = None,
                    config: Config | None = None, model: str = DEFAULT_MODEL, max_turns: int = 25,
                    log=None, llm_client: Any = None) -> list[TaskResult]:
    tasks = [t for t in TASKS if not task_ids or t.id in task_ids]
    base = Path(base or tempfile.mkdtemp(prefix="flux-mcp-evals-"))
    base.mkdir(parents=True, exist_ok=True)
    cfg = config or Config()
    cfg.workdir = base
    session = Session(cfg)
    server = build_server(session)
    results: list[TaskResult] = []
    llm = _LLM(model, llm_client) if mode == "llm" else None
    async with Client(server) as client:
        tool_defs = await _tool_defs(client) if llm else None
        for task in tasks:
            t0 = time.monotonic()
            res = TaskResult(task.id, False)
            try:
                ctx = await _prepare(task, client, session, base)
                if llm is None:
                    await task.reference(ctx)
                else:
                    res.model, res.turns = await llm.drive(client, tool_defs, task.render_prompt(ctx), ctx, max_turns)
                res.passed, res.reason = _check(task, ctx)
                res.tool_calls = list(ctx.calls)
            except Exception as e:
                res.reason = f"{type(e).__name__}: {e}"
                if os.environ.get("FLUX_MCP_EVAL_TRACE"):
                    traceback.print_exc()
            res.seconds = round(time.monotonic() - t0, 2)
            results.append(res)
            if log:
                log(f"{'PASS' if res.passed else 'FAIL'} {task.id} ({res.seconds}s, {len(res.tool_calls)} calls)"
                    + (f": {res.reason}" if res.reason else ""))
    return results


def summary(results: list[TaskResult]) -> dict:
    n = len(results)
    passed = sum(r.passed for r in results)
    return {"tasks": n, "passed": passed, "pass_rate": round(passed / n, 4) if n else 0.0,
            "results": [asdict(r) for r in results]}


async def _tool_defs(client: Client) -> list[dict]:
    tools = (await client.list_tools()).tools
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in tools]


class _LLM:
    """A manual tool-use loop over the Anthropic Messages API; MCP tools become Claude tools."""

    def __init__(self, model: str, client: Any = None):
        if client is None:
            import anthropic  # optional dependency: pip install 'flux-ai[mcp-evals]'

            # Organisation-level keys must name the workspace to bill
            ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            client = anthropic.AsyncAnthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
        self.client = client
        self.model = model

    async def drive(self, client: Client, tools: list[dict], prompt: str, ctx: Ctx, max_turns: int) -> tuple[str, int]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        served = self.model
        for turn in range(1, max_turns + 1):
            resp = await self.client.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, tools=tools, messages=messages,
                output_config={"effort": "high"},
            )
            served = resp.model
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                raise RuntimeError("the model refused the request")
            if resp.stop_reason != "tool_use":
                return served, turn
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                r = await client.call_tool(block.name, dict(block.input))
                ctx.calls.append(block.name)
                text = "\n".join(getattr(c, "text", "") for c in r.content) or "{}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text,
                                "is_error": bool(r.is_error)})
            messages.append({"role": "user", "content": results})
        return served, max_turns


def write_report(path: Path, mode: str, results: list[TaskResult], model: str | None = None) -> dict:
    rep = {"mode": mode, "model": model if mode == "llm" else None, **summary(results)}
    path.write_text(json.dumps(rep, indent=1))
    return rep
