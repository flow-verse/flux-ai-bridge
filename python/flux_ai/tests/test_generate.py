import io
import json
from pathlib import Path

import numpy as np
import pytest

from flux_ai import backends, cli, licences, music, provenance, serve
from flux_ai.backends import Backend, GenerateRequest, GenerateResult, get_backend, list_backends


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUX_AI_CACHE", str(tmp_path / "cache"))
    yield


def events(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.splitlines()]


# --- interface ---------------------------------------------------------------------------

def test_every_registered_backend_satisfies_protocol():
    for name in backends.backend_names():
        b = get_backend(name)
        assert isinstance(b, Backend)
        assert b.name == name
        assert b.modes <= backends.MODES
        ok, reason = b.available()
        assert isinstance(ok, bool) and isinstance(reason, str)


def test_get_backend_is_cached_and_rejects_unknown():
    assert get_backend("sine-test") is get_backend("sine-test")
    with pytest.raises(KeyError):
        get_backend("nope")


def test_list_backends_fields():
    bl = {b["name"]: b for b in list_backends()}
    assert set(bl) == {"ace-step", "magenta-rt", "stable-audio-3", "amt", "sine-test"}
    assert bl["ace-step"]["default"] and bl["magenta-rt"]["default"]
    assert bl["stable-audio-3"]["opt_in"] and not bl["stable-audio-3"]["default"]
    assert bl["sine-test"]["available"]
    assert bl["ace-step"]["licence"] == "MIT"
    assert "1M" in bl["stable-audio-3"]["commercial_use"]
    assert bl["amt"]["output"] == "midi"


def test_sine_test_is_deterministic_and_on_tempo(tmp_path):
    b = get_backend("sine-test")
    seen = []
    r1 = b.generate(GenerateRequest("x", 4.0, tmp_path / "a.wav", seed=3, bpm=122, key="A minor"), progress=lambda f, m: seen.append(f))
    r2 = b.generate(GenerateRequest("x", 4.0, tmp_path / "b.wav", seed=3, bpm=122, key="A minor"))
    r3 = b.generate(GenerateRequest("x", 4.0, tmp_path / "c.wav", seed=4, bpm=122, key="A minor"))
    assert isinstance(r1, GenerateResult)
    assert r1.seconds == pytest.approx(4.0, abs=1e-3) and r1.sample_rate == 44100
    a, _ = music.read_wav(r1.audio)
    assert np.array_equal(a, music.read_wav(r2.audio)[0])
    assert not np.array_equal(a, music.read_wav(r3.audio)[0])
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
    # kick onsets every 60/122 s: energy envelope autocorrelation peaks at the beat period
    env = np.abs(a[:, 0]).reshape(-1, 441).mean(axis=1)  # 10 ms hops
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lag = np.argmax(ac[30:80]) + 30
    assert abs(lag * 0.01 - 60 / 122) < 0.015


def test_sine_test_continue_and_repaint(tmp_path):
    b = get_backend("sine-test")
    ctx = b.generate(GenerateRequest("x", 2.0, tmp_path / "ctx.wav", bpm=120, key="C")).audio
    cont = b.generate(GenerateRequest("x", 3.0, tmp_path / "cont.wav", bpm=120, key="C", mode="continue", context_audio=ctx))
    assert cont.seconds == pytest.approx(5.0, abs=1e-3)
    rep = b.generate(GenerateRequest("x", 1.0, tmp_path / "rep.wav", seed=9, bpm=120, key="C", mode="repaint",
                                     context_audio=ctx, extra={"start": 0.5, "end": 1.0}))
    x, y = music.read_wav(ctx)[0], music.read_wav(rep.audio)[0]
    assert np.array_equal(x[:22050], y[:22050]) and not np.array_equal(x, y)


def test_mode_validation(tmp_path):
    b = get_backend("sine-test")
    with pytest.raises(ValueError, match="context"):
        b.generate(GenerateRequest("x", 1.0, tmp_path / "a.wav", mode="continue"))
    with pytest.raises(ValueError, match="mode"):
        b.generate(GenerateRequest("x", 1.0, tmp_path / "a.wav", mode="remix"))


def test_magenta_note_guide_marks_beats_and_key():
    from flux_ai.backends.magenta_rt import FRAME_RATE, note_guide

    rows = note_guide(frames=FRAME_RATE * 8, bpm=122, key="A minor")
    onset_frames = [f for f, r in enumerate(rows) if 2 in r]
    beat = 60 / 122 * FRAME_RATE
    assert len(onset_frames) == int(8 * 122 / 60) + 1
    assert all(abs(f - round(i * beat)) == 0 for i, f in enumerate(onset_frames))
    first = rows[0]
    assert {p % 12 for p, s in enumerate(first) if s == 2} == {9, 0, 4}  # A minor triad
    assert first.count(-1) == 128 - 4  # everything else left to the model
    strict = note_guide(frames=10, bpm=122, key="A minor", strict=True)
    assert -1 not in strict[0]


def test_key_parsing():
    assert music.parse_key("A minor") == (9, "minor")
    assert music.parse_key("Am") == (9, "minor")
    assert music.parse_key("Bb maj") == (10, "major")
    assert music.parse_key("F#") == (6, "major")
    assert music.normalise_key("eb min") == "D# minor"
    with pytest.raises(ValueError):
        music.parse_key("H dorian")


# --- provenance --------------------------------------------------------------------------

def test_provenance_sidecar(tmp_path):
    out = tmp_path / "song.wav"
    r = get_backend("sine-test").generate(GenerateRequest("dark techno", 2.0, out, seed=7, bpm=122, key="A minor", extra={"x": 1}))
    side = tmp_path / "song.wav.provenance.json"
    assert side.is_file() and r.provenance["sidecar"] == str(side)
    rec = json.loads(side.read_text())
    assert rec["ai_generated"] is True
    assert rec["backend"] == "sine-test" and rec["model"]["id"] == "sine-test-1"
    assert rec["prompt"] == "dark techno" and rec["seed"] == 7
    assert rec["conditioning"]["bpm"] == 122 and rec["conditioning"]["key"] == "A minor"
    assert rec["conditioning"]["extra"] == {"x": 1}
    assert rec["output"]["sha256"] == provenance.sha256(out)
    assert rec["flux_ai_version"]
    for k in ("started_at", "finished_at", "model"):
        assert rec[k]
    assert "weights_licence" in rec["model"]


# --- licence gating ----------------------------------------------------------------------

class GatedBackend:
    """A weights-bearing fake: gated on the AMT licence id, renders via sine-test."""
    name = "gated-fake"
    licence_id = "amt"
    modes = frozenset({"generate"})

    def available(self):
        return True, "fake"

    def generate(self, req, progress=None):
        licences.require(self.licence_id)
        r = get_backend("sine-test").generate(req, progress)
        r.provenance["model"]["licence_acceptance"] = licences.accepted(self.licence_id)
        return r


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setitem(backends._REGISTRY, "gated-fake", ("x", "x", "fake", False, True))
    monkeypatch.setitem(backends._INSTANCES, "gated-fake", GatedBackend())


def test_licence_required_then_recorded_once(tmp_path, gated, capsys):
    argv = ["--json", "generate", "--backend", "gated-fake", "--prompt", "p", "--seconds", "1", "--out", str(tmp_path / "o.wav")]
    assert cli.main(argv) == 1
    ev = events(capsys)
    assert ev[-1]["event"] == "error" and "--accept-licence amt" in ev[-1]["message"]
    assert ev[-1]["error_type"] == "LicenceNotAccepted"
    assert not licences.licence_file().exists()

    assert cli.main(argv + ["--accept-licence", "amt"]) == 0
    assert events(capsys)[-1]["event"] == "done"
    rec = json.loads(licences.licence_file().read_text())["amt"]
    assert rec["licence"] == "Apache-2.0" and rec["via"] == "--accept-licence"

    assert cli.main(argv) == 0  # remembered
    assert events(capsys)[-1]["event"] == "done"


def test_unknown_licence_id_is_an_error(tmp_path, capsys):
    rc = cli.main(["--json", "generate", "--backend", "sine-test", "--prompt", "p", "--seconds", "1",
                   "--out", str(tmp_path / "o.wav"), "--accept-licence", "bogus"])
    assert rc == 1 and "unknown licence id" in events(capsys)[-1]["message"]


def test_heavy_backends_refuse_without_licence(tmp_path):
    for name in ("ace-step", "magenta-rt", "stable-audio-3", "amt"):
        b = get_backend(name)
        with pytest.raises(licences.LicenceNotAccepted):
            b.generate(GenerateRequest("p", 1.0, tmp_path / "o.wav"))


# --- CLI protocol v2 ---------------------------------------------------------------------

def test_generate_json_events_v2(tmp_path, capsys):
    out = tmp_path / "g.wav"
    rc = cli.main(["--json", "--job-id", "job-1", "generate", "--backend", "sine-test", "--prompt", "house",
                   "--seconds", "2", "--bpm", "122", "--key", "A minor", "--seed", "5", "--out", str(out)])
    assert rc == 0
    ev = events(capsys)
    kinds = [e["event"] for e in ev]
    assert kinds[0] == "started" and kinds[-1] == "done" and "progress" in kinds
    for e in ev:
        assert e["protocol_version"] == 2 and e["job_id"] == "job-1" and e["job"] == "generate"
    # v1 fields that worker.rs deserialises
    assert isinstance(ev[0]["input"], str)
    done = ev[-1]
    assert done["outputs"] == [str(out), str(out) + ".provenance.json"]
    assert done["backend"] == "sine-test" and done["seconds"] == pytest.approx(2.0)
    fr = [e["fraction"] for e in ev if e["event"] == "progress"]
    assert fr == sorted(fr) and all(0 <= f <= 1 for f in fr)
    assert all(isinstance(e["message"], str) for e in ev if e["event"] == "progress")


def test_v1_events_still_parse_like_worker_rs(tmp_path, capsys):
    """worker.rs: serde tag="event", started{job,input}, done{job,outputs}, error{job,message}; unknown fields ignored."""
    cli.main(["--json", "generate", "--backend", "sine-test", "--prompt", "p", "--seconds", "1", "--out", str(tmp_path / "a.wav")])
    cli.main(["--json", "generate", "--backend", "nope", "--prompt", "p", "--out", str(tmp_path / "b.wav")])
    required = {"started": {"job": str, "input": str}, "done": {"job": str, "outputs": list}, "error": {"job": str, "message": str}}
    for e in events(capsys):
        if e["event"] in required:
            for k, t in required[e["event"]].items():
                assert isinstance(e[k], t)
            if e["event"] == "done":
                assert all(isinstance(o, str) for o in e["outputs"])


def test_backends_command(capsys):
    assert cli.main(["--json", "backends"]) == 0
    ev = events(capsys)
    assert [e["event"] for e in ev] == ["started", "done"]
    assert ev[-1]["outputs"] == [] and {b["name"] for b in ev[-1]["backends"]} >= {"ace-step", "sine-test"}
    assert cli.main(["backends"]) == 0
    assert "sine-test" in capsys.readouterr().out


def test_generate_human_mode(tmp_path, capsys):
    assert cli.main(["generate", "--backend", "sine-test", "--prompt", "p", "--seconds", "1", "--out", str(tmp_path / "h.wav")]) == 0
    assert "h.wav" in capsys.readouterr().out


def test_extra_and_model_flags(tmp_path, capsys):
    out = tmp_path / "e.wav"
    cli.main(["--json", "generate", "--backend", "sine-test", "--prompt", "p", "--seconds", "1", "--model", "m",
              "--extra", "steps=12", "--extra", "name=abc", "--out", str(out)])
    rec = json.loads((tmp_path / "e.wav.provenance.json").read_text())
    assert rec["conditioning"]["extra"] == {"steps": 12, "name": "abc", "model": "m"}


# --- serve -------------------------------------------------------------------------------

def test_serve_runs_jobs_and_keeps_going_after_errors(tmp_path, capsys):
    reqs = [
        {"job": "ping"},
        {"job_id": "a", "job": "generate", "params": {"backend": "sine-test", "prompt": "p", "seconds": 1, "out": str(tmp_path / "1.wav")}},
        "not json",
        {"job_id": "b", "job": "generate", "params": {"backend": "nope", "prompt": "p", "out": str(tmp_path / "2.wav")}},
        {"job_id": "c", "job": "backends"},
        {"job_id": "d", "job": "generate", "params": {"backend": "sine-test", "prompt": "p", "seconds": 1, "out": str(tmp_path / "3.wav")}},
        {"job": "shutdown"},
        {"job_id": "never", "job": "backends"},
    ]
    stdin = io.StringIO("\n".join(r if isinstance(r, str) else json.dumps(r) for r in reqs) + "\n")
    assert serve.main(stdin=stdin) == 0
    ev = events(capsys)
    assert ev[0]["event"] == "ready" and ev[1]["event"] == "pong"
    assert all(e["protocol_version"] == 2 for e in ev)
    by = {}
    for e in ev:
        by.setdefault(e["job_id"], []).append(e["event"])
    assert by["a"][0] == "started" and by["a"][-1] == "done"
    assert by["b"][-1] == "error"
    assert by["c"] == ["started", "done"]
    assert by["d"][-1] == "done"
    assert "never" not in by
    assert ev[-1]["event"] == "bye"
    assert any(e["event"] == "error" and "bad request" in e["message"] for e in ev)
    assert (tmp_path / "3.wav.provenance.json").exists()


def test_serve_cli_entry(tmp_path, monkeypatch, capsys):
    for argv in (["--json", "serve"], ["serve", "--json"]):
        monkeypatch.setattr("sys.stdin", io.StringIO('{"job":"shutdown"}\n'))
        assert cli.main(argv) == 0
        assert [e["event"] for e in events(capsys)] == ["ready", "bye"]


def test_json_flag_after_subcommand(tmp_path, capsys):
    out = tmp_path / "t.wav"
    assert cli.main(["generate", "--backend", "sine-test", "--prompt", "p", "--seconds", "1", "--out", str(out),
                     "--json", "--job-id", "late"]) == 0
    ev = events(capsys)
    assert ev[-1]["event"] == "done" and all(e["job_id"] == "late" for e in ev)


# --- end to end in a real process -------------------------------------------------------

def test_stdout_is_protocol_only_in_a_real_process(tmp_path):
    """C-level and Python prints during a job must not reach stdout in JSON mode."""
    import os
    import subprocess
    import sys

    code = (
        "import os, sys, flux_ai.backends.sine_test as s\n"
        "orig = s.SineTestBackend.generate\n"
        "def noisy(self, req, progress=None):\n"
        "    print('python noise'); os.write(1, b'fd noise\\n')\n"
        "    return orig(self, req, progress)\n"
        "s.SineTestBackend.generate = noisy\n"
        "from flux_ai import cli\n"
        f"sys.exit(cli.main(['--json', 'generate', '--backend', 'sine-test', '--prompt', 'p', '--seconds', '1', '--out', {str(tmp_path / 'x.wav')!r}]))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "FLUX_AI_CACHE": str(tmp_path / "c")}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert lines and all(json.loads(l)["protocol_version"] == 2 for l in lines)
    assert "python noise" in p.stderr and "fd noise" in p.stderr


def test_bench_sine_test(tmp_path):
    pytest.importorskip("librosa")
    from flux_ai import bench

    assert bench.main(["--backend", "sine-test", "--seeds", "2", "--seconds", "20", "--out-dir", str(tmp_path)]) == 0
    rep = json.loads((tmp_path / "bench.json").read_text())
    s = rep["results"][0]
    assert s["n"] == 2 and s["bpm_within_1"] == 2 and s["key_match"] == 2
    assert s["compute_s_per_audio_min"] > 0 and s["peak_rss_gb"] > 0
    assert (tmp_path / "bench.md").read_text().startswith("# flux-ai bench")


def test_bench_plan_pairs_models_with_backends():
    from flux_ai.bench import _plan

    argv = ["--backend", "ace-step", "--model", "turbo", "--model", "xl-turbo", "--backend", "magenta-rt", "--backend", "amt", "--model", "small"]
    assert _plan(argv) == [("ace-step", "turbo"), ("ace-step", "xl-turbo"), ("magenta-rt", None), ("amt", "small")]
