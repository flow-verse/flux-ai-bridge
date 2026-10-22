import json
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


def test_json_mode_emits_started_and_done(tmp_path: Path, capsys):
    stems = tmp_path / "stems" / "htdemucs" / "in"
    stems.mkdir(parents=True)
    (stems / "drums.wav").touch()
    with mock.patch.object(cli.subprocess, "call", return_value=0):
        rc = cli.main(["--json", "separate", "in.wav", "--out", str(tmp_path / "stems")])
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert [l["event"] for l in lines] == ["started", "done"]
    assert lines[1]["outputs"] == [str(stems / "drums.wav")]


def test_json_mode_reports_failure(tmp_path: Path, capsys):
    with mock.patch.object(cli.subprocess, "call", return_value=2):
        rc = cli.main(["--json", "separate", "in.wav", "--out", str(tmp_path)])
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert rc == 1
    assert lines[-1]["event"] == "error"
