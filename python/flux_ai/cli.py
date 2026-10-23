"""flux-ai CLI: thin, dependency-lazy wrappers over the ML models.

With --json, events are emitted as one JSON object per line on stdout (and
nothing else), so a host process (the CLAP plugin, an MCP server) can parse
them. Protocol v2 (see docs/protocol.md) adds ``protocol_version`` and
``job_id`` to every event and a ``progress`` event; the v1 fields stay:

    {"event": "started", "protocol_version": 2, "job_id": "…", "job": "separate", "input": "..."}
    {"event": "progress", "protocol_version": 2, "job_id": "…", "job": "generate", "fraction": 0.4, "message": "..."}
    {"event": "done", "protocol_version": 2, "job_id": "…", "job": "separate", "outputs": ["...drums.wav", ...]}
    {"event": "error", "protocol_version": 2, "job_id": "…", "job": "separate", "message": "..."}
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from flux_ai.protocol import Emitter, protocol_stdout


def emit(enabled: bool, **event) -> None:
    """v1 helper kept for callers that import it; new code uses protocol.Emitter."""
    if enabled:
        print(json.dumps(event), flush=True)


def separate(src: Path, out: Path, model: str = "htdemucs", quiet: bool = False) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out), str(src)]
    # In JSON mode keep demucs' progress bars off stdout so stdout stays parseable.
    rc = subprocess.call(cmd, stdout=sys.stderr if quiet else None)
    if rc != 0:
        raise RuntimeError(f"demucs exited with {rc}")
    return sorted((out / model / src.stem).glob("*.wav"))


def transcribe(src: Path, out: Path, quiet: bool = False) -> list[Path]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    out.mkdir(parents=True, exist_ok=True)
    predict_and_save([str(src)], str(out), True, False, False, False, ICASSP_2022_MODEL_PATH)
    return sorted(out.glob(f"{src.stem}*.mid"))


JOBS = {"separate": separate, "transcribe": transcribe}


def _parse_extra(items: list[str] | None) -> dict:
    extra = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"--extra expects KEY=VALUE, got '{item}'")
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v
    return extra


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flux-ai")
    p.add_argument("--json", action="store_true", help="emit JSON-lines protocol events on stdout")
    p.add_argument("--job-id", help="job id to put on every event (default: random)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in JOBS:
        s = sub.add_parser(name)
        s.add_argument("input", type=Path)
        s.add_argument("--out", type=Path, required=True)

    g = sub.add_parser("generate", help="generate audio (or MIDI) with a generation backend")
    g.add_argument("--backend", default="ace-step", help="ace-step | magenta-rt | stable-audio-3 | amt | sine-test")
    g.add_argument("--prompt", required=True)
    g.add_argument("--seconds", type=float, default=30.0)
    g.add_argument("--bpm", type=float)
    g.add_argument("--key", help='e.g. "A minor", "F# major", "Bbm"')
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--mode", default="generate", choices=["generate", "continue", "repaint", "infill"])
    g.add_argument("--context", type=Path, help="context audio/MIDI for continue/repaint/infill")
    g.add_argument("--start", type=float, help="repaint/infill region start (s)")
    g.add_argument("--end", type=float, help="repaint/infill region end (s)")
    g.add_argument("--lyrics")
    g.add_argument("--vocals", action="store_true", help="allow vocals (default: instrumental)")
    g.add_argument("--model", help="model tier, e.g. turbo|xl-turbo (ace-step), base|small (magenta-rt)")
    g.add_argument("--extra", action="append", metavar="KEY=VALUE", help="backend-specific option (JSON value)")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--accept-licence", action="append", default=[], metavar="ID",
                   help="accept a model licence non-interactively (recorded once)")

    b = sub.add_parser("backends", help="list generation backends, availability and licences")
    b.add_argument("--accept-licence", action="append", default=[], metavar="ID")

    sv = sub.add_parser("serve", help="warm worker: read JSON job requests from stdin, one per line")
    sv.add_argument("--accept-licence", action="append", default=[], metavar="ID")

    bn = sub.add_parser("bench", help="benchmark generation backends (see flux_ai.bench)")
    bn.add_argument("bench_args", nargs=argparse.REMAINDER)
    return p


def generate_params(a: argparse.Namespace) -> dict:
    extra = _parse_extra(a.extra)
    if a.model:
        extra["model"] = a.model
    if a.start is not None:
        extra["start"] = a.start
    if a.end is not None:
        extra["end"] = a.end
    return {
        "backend": a.backend, "prompt": a.prompt, "seconds": a.seconds, "bpm": a.bpm, "key": a.key,
        "seed": a.seed, "mode": a.mode, "context": str(a.context) if a.context else None,
        "lyrics": a.lyrics, "instrumental": not a.vocals, "extra": extra, "out": str(a.out),
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "bench" in argv and set(argv[: argv.index("bench")]) <= {"--json"}:
        from flux_ai import bench  # bench has its own option parser

        return bench.main(argv[argv.index("bench") + 1:])
    a = build_parser().parse_args(argv)
    from flux_ai import jobs

    if a.cmd == "serve":
        from flux_ai import serve

        return serve.main(json_mode=a.json, accept=a.accept_licence)

    with protocol_stdout(a.json) as out:
        em = Emitter(a.json, a.cmd, a.job_id, stream=out)
        if a.cmd in JOBS:
            params = {"input": str(a.input), "out": str(a.out)}
        elif a.cmd == "generate":
            params = generate_params(a)
        else:
            params = {}
        try:
            result = jobs.run(a.cmd, params, em, accept=getattr(a, "accept_licence", []), interactive=not a.json)
        except Exception as e:  # report to the host instead of a bare traceback
            em.error(str(e), error_type=type(e).__name__)
            if not a.json:
                raise
            return 1
    if not a.json:
        jobs.print_human(a.cmd, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
