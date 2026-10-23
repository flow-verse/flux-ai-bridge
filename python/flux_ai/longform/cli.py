# SPDX-License-Identifier: GPL-3.0-or-later
"""`flux-ai longform` subcommand.

    flux-ai longform --genre "deep house" --hours 1 --out runs/dh1
    flux-ai longform --genre techno --hours 4 --plan-only --out runs/t4
    flux-ai longform --plan runs/t4/plan.json --out runs/t4 --json

With --json, stdout carries only JSON lines in job protocol v2:
    {"event": "started",  "protocol_version": 2, "job": "longform", "job_id": "...", ...}
    {"event": "progress", "protocol_version": 2, "job": "longform", "job_id": "...", "stage": "render", "fraction": 0.4, "message": "s012"}
    {"event": "done",     "protocol_version": 2, "job": "longform", "job_id": "...", "outputs": [...], "report": {...}}
    {"event": "error",    "protocol_version": 2, "job": "longform", "job_id": "...", "message": "..."}
v1 consumers keep working: they only read `event`, `outputs` and `message`.
This module imports nothing heavy at import time (it is loaded by flux_ai.cli).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

PROTOCOL_VERSION = 2


def register(sub) -> None:
    p = sub.add_parser("longform", help="plan and render N hours of a genre (P5 long-form orchestrator)")
    p.add_argument("--genre", default=None, help='genre table name, e.g. "deep house", techno, lofi, ambient, maqam')
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plan", type=Path, default=None, help="execute an existing (e.g. agent-written) plan JSON")
    p.add_argument("--plan-only", action="store_true", help="write plan.json and stop")
    p.add_argument("--execute", action="store_true", help="render the plan (default unless --plan-only)")
    p.add_argument("--neural-backend", default=None, help="add a neural layer via flux_ai.backends (e.g. ace-step)")
    p.add_argument("--neural-sections", default="breakdown", help="comma-separated section types for the neural layer, or 'all'")
    p.add_argument("--out", type=Path, default=Path("longform-out"))
    p.add_argument("--format", default="auto", choices=["auto", "wav", "flac", "rf64"],
                   help="auto: 24-bit WAV below 4 h, FLAC from 4 h")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--renderer", default="builtin", choices=["builtin", "flux-host"])
    p.add_argument("--json", dest="longform_json", action="store_true", help="JSON-lines events (job protocol v2)")
    p.set_defaults(func=run_cli)


def run_cli(a: argparse.Namespace) -> int:
    use_json = bool(getattr(a, "json", False) or getattr(a, "longform_json", False))
    job_id = uuid.uuid4().hex[:12]

    def emit(event: str, **kw) -> None:
        if use_json:
            print(json.dumps({"event": event, "protocol_version": PROTOCOL_VERSION, "job": "longform",
                              "job_id": job_id, **kw}), flush=True)
        elif event == "progress":
            print(f"[{kw.get('stage')}] {kw.get('fraction', 0) * 100:5.1f}% {kw.get('message', '')}", file=sys.stderr, flush=True)

    emit("started", input=str(a.plan) if a.plan else a.genre, hours=a.hours, seed=a.seed, out=str(a.out))
    try:
        from . import pipeline, planner
        from .render import get_renderer

        if a.plan:
            sp = planner.SetPlan.from_file(a.plan)
        else:
            if not a.genre:
                raise ValueError("--genre or --plan is required")
            cons = {}
            if a.neural_backend:
                cons["neural"] = {"backend": a.neural_backend, "types": a.neural_sections.split(",")}
            sp = planner.plan(a.genre, a.hours, a.seed, cons)
        if a.neural_backend and a.plan:
            for s in sp.data["sections"]:
                if s["type"] in a.neural_sections.split(",") or a.neural_sections == "all":
                    s["neural"] = s.get("neural") or {"backend": a.neural_backend, "prompt": f"{s['genre']} instrumental",
                                                      "gain_db": -10.0, "seed": s["seeds"]["section"]}
        a.out.mkdir(parents=True, exist_ok=True)
        plan_path = a.out / "plan.json"
        if a.plan_only:
            sp.to_json(plan_path)
            emit("done", outputs=[str(plan_path)], sections=len(sp.sections),
                 planned_seconds=round(sp.duration_seconds(), 2),
                 no_repeat_violations=len(planner.no_repeat_violations(sp.data)))
            if not use_json:
                print(plan_path)
            return 0
        renderer = get_renderer(a.renderer)
        if a.renderer == "flux-host":
            ok, why = renderer.available()
            if not ok:
                raise RuntimeError(f"flux-host renderer unavailable: {why}")
        report = pipeline.run(sp, a.out, renderer=renderer, workers=a.workers, fmt=a.format,
                              progress=lambda stage, frac, msg="": emit("progress", stage=stage, fraction=round(frac, 4), message=msg))
    except Exception as e:
        emit("error", message=f"{type(e).__name__}: {e}")
        if not use_json:
            raise
        return 1
    outs = report["outputs"]
    emit("done", outputs=[outs["master"], outs["cue"], outs["chapters"], outs["provenance"], outs["report"]], report=report)
    if not use_json:
        print(json.dumps({k: report[k] for k in ("audio_hms", "wall_clock_seconds", "speed_factor", "loudness_lufs",
                                                 "true_peak_dbtp", "join_clicks")} | {"master": outs["master"]}, indent=1))
    return 0
