# SPDX-License-Identifier: GPL-3.0-or-later
"""python -m flux_mcp.evals [--llm] [--model ID] [--task ID ...] [--json report.json]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anyio

from flux_mcp.evals.runner import DEFAULT_MODEL, run_suite, summary, write_report
from flux_mcp.evals.tasks import TASKS
from flux_mcp.server import Config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flux_mcp.evals")
    p.add_argument("--llm", action="store_true", help="let a model drive the tools (needs Anthropic credentials)")
    p.add_argument("--model", default=os.environ.get("FLUX_MCP_EVAL_MODEL", DEFAULT_MODEL))
    p.add_argument("--task", action="append", help="run only these task ids (repeatable)")
    p.add_argument("--engine", help="flux-engine path (default: $FLUX_ENGINE, then PATH)")
    p.add_argument("--plugin-path", action="append", default=[], help="extra CLAP folder for the engine")
    p.add_argument("--out", type=Path, help="scratch directory (default: a new temp dir)")
    p.add_argument("--json", type=Path, help="write the report here")
    p.add_argument("--max-turns", type=int, default=25)
    p.add_argument("--list", action="store_true", help="list the tasks and exit")
    a = p.parse_args(argv)
    if a.list:
        for t in TASKS:
            print(f"{t.id:28} {t.prompt}")
        return 0
    if a.llm and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("LLM runner skipped: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) to run it", file=sys.stderr)
        return 0
    engine_args = [x for d in a.plugin_path for x in ("--plugin-path", d)]
    cfg = Config(engine=a.engine, engine_args=engine_args)
    mode = "llm" if a.llm else "scripted"
    results = anyio.run(lambda: run_suite(mode, a.task, a.out, cfg, a.model, a.max_turns,
                                          log=lambda m: print(m, file=sys.stderr, flush=True)))
    rep = write_report(a.json, mode, results, a.model) if a.json else {"mode": mode, **summary(results)}
    print(f"{mode}: {rep['passed']}/{rep['tasks']} passed ({rep['pass_rate'] * 100:.1f}%)", file=sys.stderr)
    print(json.dumps({k: v for k, v in rep.items() if k != "results"}))
    return 0 if rep["passed"] == rep["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
