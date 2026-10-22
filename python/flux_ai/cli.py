"""flux-ai CLI: thin, dependency-lazy wrappers over the ML models."""
import argparse
import subprocess
import sys
from pathlib import Path


def separate(src: Path, out: Path, model: str = "htdemucs") -> int:
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.call([sys.executable, "-m", "demucs", "-n", model, "-o", str(out), str(src)])


def transcribe(src: Path, out: Path) -> int:
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    out.mkdir(parents=True, exist_ok=True)
    predict_and_save([str(src)], str(out), True, False, False, False, ICASSP_2022_MODEL_PATH)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flux-ai")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("separate", "transcribe"):
        s = sub.add_parser(name)
        s.add_argument("input", type=Path)
        s.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    return {"separate": separate, "transcribe": transcribe}[a.cmd](a.input, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
