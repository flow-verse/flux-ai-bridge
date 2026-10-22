from pathlib import Path
from unittest import mock

from flux_ai import cli


def test_help_exits():
    try:
        cli.main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_separate_invokes_demucs_with_model_and_out(tmp_path: Path):
    with mock.patch.object(cli.subprocess, "call", return_value=0) as call:
        rc = cli.main(["separate", "in.wav", "--out", str(tmp_path / "stems")])
    assert rc == 0
    args = call.call_args[0][0]
    assert args[1:3] == ["-m", "demucs"]
    assert "htdemucs" in args and str(tmp_path / "stems") in args
    assert (tmp_path / "stems").is_dir()
