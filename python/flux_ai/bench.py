"""Benchmark generation backends: speed, memory and BPM/key adherence.

    flux-ai bench --backend ace-step --model turbo --model xl-turbo \\
        --backend magenta-rt --model base --model small \\
        --seeds 5 --seconds 120 --bpm 122 --key "A minor" --out-dir bench/ \\
        --accept-licence ace-step-1.5 --accept-licence magenta-rt-2

Each (backend, model tier) runs in its own subprocess so peak memory is per
tier. The first seed includes model loading (cold); compute rate is the median
of the remaining (warm) seeds. Outputs ``bench.json`` and ``bench.md`` (plus
the generated audio and provenance sidecars) in ``--out-dir``.

``--model`` applies to the most recent ``--backend``; without one the
backend's default tier is used.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = "deep house, warm analog pads, punchy four-on-the-floor kick, rolling bassline, instrumental"


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _accel_peaks() -> dict:
    out = {}
    if "mlx.core" in sys.modules:
        mx = sys.modules["mlx.core"]
        for fn in ("get_peak_memory",):
            f = getattr(mx, fn, None) or getattr(getattr(mx, "metal", None), fn, None)
            if f:
                try:
                    out["mlx_peak_bytes"] = int(f())
                except Exception:
                    pass
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        try:
            if torch.backends.mps.is_available():
                out["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
        except Exception:
            pass
    return out


def worker(args: argparse.Namespace) -> int:
    """Run all seeds for one (backend, tier) in this process; print one JSON line."""
    from flux_ai import licences
    from flux_ai.backends import GenerateRequest, get_backend

    for lid in args.accept_licence:
        licences.accept(lid, via="bench --accept-licence")
    b = get_backend(args.backend)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = json.loads(args.extra_json or "{}")
    if args.model:
        extra["model"] = args.model
    runs = []
    for seed in args.seed_list:
        out = out_dir / f"{args.backend}-{args.model or 'default'}-seed{seed}.wav"
        req = GenerateRequest(prompt=args.prompt, seconds=args.seconds, out=out, seed=seed, bpm=args.bpm, key=args.key, extra=dict(extra))
        t0 = time.perf_counter()
        res = b.generate(req)
        dt = time.perf_counter() - t0
        runs.append({"seed": seed, "wall_s": round(dt, 3), "audio": str(res.audio), "audio_seconds": res.seconds,
                     "model": res.model, "sample_rate": res.sample_rate})
        print(f"[bench] {args.backend}/{args.model} seed {seed}: {dt:.1f}s for {res.seconds:.1f}s audio", file=sys.stderr, flush=True)
    rec = {"backend": args.backend, "tier": args.model or "default", "runs": runs,
           "peak_rss_bytes": _peak_rss_bytes(), **_accel_peaks()}
    print("BENCH_RESULT " + json.dumps(rec), flush=True)
    return 0


def _versions() -> dict:
    from importlib.metadata import version

    out = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for p in ("flux-ai", "ace-step", "magenta-rt", "mlx", "torch", "transformers", "librosa", "numpy", "stable-audio-3", "anticipation"):
        try:
            out[p] = version(p)
        except Exception:
            pass
    try:
        out["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        out["memory_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    return out


def summarise(rec: dict, bpm: float | None, key: str | None) -> dict:
    from flux_ai.analysis import analyse

    for r in rec["runs"]:
        if r["sample_rate"]:
            r["analysis"] = analyse(Path(r["audio"]), bpm, key)
    walls = [r["wall_s"] for r in rec["runs"]]
    secs = [r["audio_seconds"] for r in rec["runs"]]
    warm = walls[1:] or walls
    warm_secs = secs[1:] or secs
    per_min = [w / (s / 60.0) for w, s in zip(warm, warm_secs)]
    an = [r["analysis"] for r in rec["runs"] if "analysis" in r]
    s = {
        "backend": rec["backend"], "tier": rec["tier"], "model": rec["runs"][0]["model"] if rec["runs"] else None,
        "n": len(walls), "audio_seconds": round(statistics.mean(secs), 2) if secs else None,
        "cold_wall_s": walls[0] if walls else None,
        "warm_wall_s_median": round(statistics.median(warm), 2) if warm else None,
        "compute_s_per_audio_min": round(statistics.median(per_min), 2) if per_min else None,
        "realtime_factor": round(statistics.median(w / s for w, s in zip(warm, warm_secs)), 3) if warm else None,
        "peak_rss_gb": round(rec["peak_rss_bytes"] / 2**30, 2),
        "mlx_peak_gb": round(rec["mlx_peak_bytes"] / 2**30, 2) if "mlx_peak_bytes" in rec else None,
        "mps_driver_gb": round(rec["mps_driver_allocated_bytes"] / 2**30, 2) if "mps_driver_allocated_bytes" in rec else None,
    }
    if an and bpm:
        s["bpm_est"] = [a["bpm_est"] for a in an]
        s["bpm_est_flat_prior"] = [a["bpm_est_flat_prior"] for a in an]
        s["bpm_within_1"] = sum(a["bpm_within_1"] for a in an)
        s["bpm_within_1_flat_prior"] = sum(a["bpm_within_1_flat_prior"] for a in an)
        s["bpm_within_1_octave_ok"] = sum(a["bpm_within_1_octave_ok"] for a in an)
    if an and key:
        s["key_est"] = [a["key_est"] for a in an]
        s["key_match"] = sum(a["key_match"] for a in an)
        s["key_match_or_relative"] = sum(a["key_match_or_relative"] for a in an)
    return s


def markdown(report: dict) -> str:
    c = report["config"]
    lines = [
        f"# flux-ai bench: {c['seconds']:g} s, {c['bpm']} BPM, {c['key']}",
        "",
        f"Prompt: \"{c['prompt']}\"; seeds {c['seeds']}. Generated {report['finished_at']}.",
        "",
        "| backend | tier | n | cold s | warm s (median) | s compute / audio min | RTF | peak RSS GB | MLX peak GB | BPM within ±1 | key match (exact / incl. relative) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["results"]:
        if "error" in s:
            lines.append(f"| {s['backend']} | {s['tier']} | 0 | failed: {s['error'][:120]} |||||||||")
            continue
        n = s["n"]
        bpm = f"{s.get('bpm_within_1', '-')}/{n}" if "bpm_within_1" in s else "-"
        key = f"{s.get('key_match', '-')}/{n} / {s.get('key_match_or_relative', '-')}/{n}" if "key_match" in s else "-"
        lines.append(
            f"| {s['backend']} | {s['tier']} | {n} | {s['cold_wall_s']} | {s['warm_wall_s_median']} | "
            f"{s['compute_s_per_audio_min']} | {s['realtime_factor']} | {s['peak_rss_gb']} | {s['mlx_peak_gb'] or '-'} | {bpm} | {key} |"
        )
    lines += ["", "Per-seed estimates:", ""]
    for s in report["results"]:
        if "bpm_est" in s:
            lines.append(f"- {s['backend']}/{s['tier']}: BPM {s['bpm_est']} (flat prior {s['bpm_est_flat_prior']}); key {s.get('key_est')}")
    lines += ["", "Versions: " + ", ".join(f"{k} {v}" for k, v in report["versions"].items()), ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flux-ai bench")
    p.add_argument("--backend", action="append", default=[], help="repeatable; --model after it picks tiers")
    p.add_argument("--model", action="append", default=[], help="tier for the preceding --backend (repeatable)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--bpm", type=float, default=122.0)
    p.add_argument("--key", default="A minor")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--extra", action="append", metavar="KEY=VALUE")
    p.add_argument("--out-dir", type=Path, default=Path("bench-out"))
    p.add_argument("--accept-licence", action="append", default=[])
    p.add_argument("--timeout", type=float, default=6 * 3600)
    # internal
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed-list", type=int, nargs="*", help=argparse.SUPPRESS)
    p.add_argument("--extra-json", help=argparse.SUPPRESS)
    return p


def _plan(argv: list[str]) -> list[tuple[str, str | None]]:
    """Pair each --model with the --backend before it."""
    plan, cur = [], None
    it = iter(argv)
    for tok in it:
        if tok == "--backend" or tok.startswith("--backend="):
            cur = tok.split("=", 1)[1] if "=" in tok else next(it)
            plan.append([cur, []])
        elif tok == "--model" or tok.startswith("--model="):
            m = tok.split("=", 1)[1] if "=" in tok else next(it)
            if cur is None:
                raise SystemExit("--model must follow a --backend")
            plan[-1][1].append(m)
    return [(b, m) for b, ms in plan for m in (ms or [None])]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.worker:
        a.backend, a.model = a.backend[0], (a.model[0] if a.model else None)
        return worker(a)
    from flux_ai.cli import _parse_extra
    from flux_ai.provenance import now

    plan = _plan(argv) or [("sine-test", None)]
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": {"seconds": a.seconds, "bpm": a.bpm, "key": a.key, "prompt": a.prompt, "seeds": seeds,
                         "extra": _parse_extra(a.extra), "plan": plan},
              "versions": _versions(), "started_at": now(), "results": [], "raw": []}
    # make the worker import this same flux_ai package, even when another copy is installed
    pkg_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))}
    for backend, tier in plan:
        cmd = [sys.executable, "-m", "flux_ai.bench", "--worker", "--backend", backend, "--seconds", str(a.seconds),
               "--bpm", str(a.bpm), "--key", a.key, "--prompt", a.prompt, "--out-dir", str(a.out_dir),
               "--extra-json", json.dumps(_parse_extra(a.extra)), "--seed-list", *map(str, seeds)]
        if tier:
            cmd += ["--model", tier]
        for lid in a.accept_licence:
            cmd += ["--accept-licence", lid]
        print(f"[bench] {backend}/{tier or 'default'}: {len(seeds)} seeds x {a.seconds:g} s", file=sys.stderr, flush=True)
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, timeout=a.timeout, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("BENCH_RESULT ")), None)
            if p.returncode != 0 or line is None:
                raise RuntimeError(f"worker exited {p.returncode}")
            rec = json.loads(line[len("BENCH_RESULT "):])
            report["raw"].append(rec)
            report["results"].append(summarise(rec, a.bpm, a.key))
        except Exception as e:
            report["results"].append({"backend": backend, "tier": tier or "default", "error": str(e)})
    report["finished_at"] = now()
    (a.out_dir / "bench.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (a.out_dir / "bench.md").write_text(markdown(report))
    print(markdown(report))
    return 0 if all("error" not in r for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
