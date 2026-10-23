"""flux-ai CLI: thin, dependency-lazy wrappers over the ML models.

With --json, progress and results are emitted as one JSON object per line on
stdout (and nothing else), so a host process (the CLAP plugin) can parse them:

    {"event": "started", "job": "separate", "input": "..."}
    {"event": "done", "job": "separate", "outputs": ["...drums.wav", ...]}
    {"event": "error", "job": "separate", "message": "..."}
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def emit(enabled: bool, **event) -> None:
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flux-ai")
    p.add_argument("--json", action="store_true", help="emit JSON-lines events on stdout")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in JOBS:
        s = sub.add_parser(name)
        s.add_argument("input", type=Path)
        s.add_argument("--out", type=Path, required=True)
    from flux_ai.longform.cli import register as register_longform  # P5; light import

    register_longform(sub)
    a = p.parse_args(argv)
    if a.cmd == "longform":
        return a.func(a)

    emit(a.json, event="started", job=a.cmd, input=str(a.input))
    try:
        outputs = JOBS[a.cmd](a.input, a.out, quiet=a.json)
    except Exception as e:  # report to the host instead of a bare traceback
        emit(a.json, event="error", job=a.cmd, message=str(e))
        if not a.json:
            raise
        return 1
    emit(a.json, event="done", job=a.cmd, outputs=[str(o) for o in outputs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
