# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end checks on a short (about 3 minute) set. Slower than the plan tests."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

for mod in ("scipy", "soundfile", "pyloudnorm", "mido", "jsonschema", "mutagen", "yaml"):
    pytest.importorskip(mod)

import pyloudnorm  # noqa: E402
import soundfile as sf  # noqa: E402

from flux_ai.longform import analysis, master, neural, outputs, pipeline, planner  # noqa: E402
from flux_ai.longform.render import BuiltinSynthRenderer  # noqa: E402


def short_plan() -> dict:
    """Two segments with a blend transition that also changes tempo, plus a neural layer."""
    p = planner.plan("deep house", 0.05, seed=21, constraints={
        "segment_minutes": 1.6, "transition_bars": 8,
        "neural": {"backend": "sine-test", "types": ["breakdown"], "gain_db": -14}}).data
    seg1 = [s for s in p["sections"] if s["segment"] == 1]
    assert seg1, "plan should have a second segment"
    for s in seg1:
        s["bpm"] = p["sections"][0]["bpm"] + 3  # tempo ramp across the transition
    return p


class CountingRenderer(BuiltinSynthRenderer):
    calls = 0

    def render(self, section_midi, section_plan, out_wav):
        type(self).calls += 1
        return super().render(section_midi, section_plan, out_wav)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("lf") / "run"
    report = pipeline.run(short_plan(), out, workers=2)
    return out, report


def test_master_meets_loudness_and_true_peak_spec(run):
    out, rep = run
    x, sr = sf.read(rep["outputs"]["master"], dtype="float64")
    assert sr == 48000 and x.shape[1] == 2
    assert 150 < len(x) / sr < 240
    lufs = pyloudnorm.Meter(sr).integrated_loudness(x)
    assert abs(lufs + 14.0) <= 0.5
    assert master.true_peak_dbtp(x) <= -1.0
    assert rep["loudness_in_spec"] and rep["true_peak_in_spec"]
    assert sf.info(rep["outputs"]["master"]).subtype == "PCM_24"


def test_transition_joins_have_no_discontinuity(run):
    out, rep = run
    x, sr = sf.read(rep["outputs"]["master"], dtype="float64")
    assert rep["joins_checked"] >= 1 and rep["join_clicks"] == 0
    assert len(analysis.find_discontinuities(x, sr)) == 0
    # the detector itself catches a step of the size a hard cut would cause (injected where the
    # programme is smoothest; inside a noise riser a 0.2 step is, by design, not an isolated jump)
    y = x.copy()
    d = np.abs(np.diff(y, axis=0)).max(axis=1)
    local = np.convolve(d, np.ones(int(0.1 * sr)), mode="same")
    k = int(np.argmin(local[sr:-sr])) + sr
    y[k:] += 0.2
    assert k - 1 in set(analysis.find_discontinuities(y, sr).tolist())


def test_provenance_and_disclosure_tags(run):
    out, rep = run
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["ai_generated"] is True and "AI-generated" in prov["disclosure"]
    assert prov["plan"]["sections"] and prov["seeds"]["plan"] == 21
    assert "human_decisions" in prov and prov["components"][2]["name"] == "builtin-synth"
    assert any(r.get("status") == "ok" for r in prov["neural_layers"])  # sine-test layer used
    meta = sf.SoundFile(rep["outputs"]["master"]).copy_metadata()
    assert "AI_GENERATED=1" in meta["comment"]
    ixml = outputs.read_ixml(Path(rep["outputs"]["master"]))
    assert "AI_GENERATED=1" in ixml and "PROVENANCE_SHA256=" in ixml
    cue = Path(rep["outputs"]["cue"]).read_text()
    assert "INDEX 01 00:00:00" in cue and cue.count("TRACK") >= 2
    chapters = Path(rep["outputs"]["chapters"]).read_text().splitlines()
    assert chapters[0].startswith("0:00 ") and len(chapters) >= 2
    sim = json.loads((out / "similarity.json").read_text())
    assert sim["windows"] >= 3 and sim["flag_count"] == 0


def test_resume_skips_finished_sections(run, tmp_path):
    out, rep = run
    n = rep["sections"]
    CountingRenderer.calls = 0
    rep2 = pipeline.run(json.loads((out / "plan.json").read_text()), out, renderer=CountingRenderer())
    assert CountingRenderer.calls == 0 and rep2["sections_cached"] == n
    victim = sorted((out / "sections").iterdir())[1]
    shutil.rmtree(victim)
    rep3 = pipeline.run(json.loads((out / "plan.json").read_text()), out, renderer=CountingRenderer())
    assert CountingRenderer.calls == 1 and rep3["sections_rendered"] == 1
    # identical audio after resume
    a = sf.read(rep["outputs"]["master"])[0]
    assert np.allclose(a, sf.read(rep3["outputs"]["master"])[0], atol=1e-6)


def test_killed_run_resumes_without_rerendering(tmp_path):
    p = planner.plan("lofi", 0.03, seed=8).data
    n = len(p["sections"])
    CountingRenderer.calls = 0
    with pytest.raises(pipeline.RunInterrupted):
        pipeline.run(p, tmp_path, renderer=CountingRenderer(), max_renders=2)
    assert CountingRenderer.calls == 2
    CountingRenderer.calls = 0
    rep = pipeline.run(p, tmp_path, renderer=CountingRenderer())
    assert CountingRenderer.calls == n - 2 and rep["sections_cached"] == 2


def test_streaming_loudness_matches_pyloudnorm():
    rng = np.random.default_rng(0)
    sr = 48000
    t = np.arange(sr * 20) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(t)), 0.1 * rng.standard_normal(len(t))], 1)
    m = master.StreamingLoudness(sr)
    for a in range(0, len(x), 30001):
        m.update(x[a:a + 30001])
    assert abs(m.integrated() - pyloudnorm.Meter(sr).integrated_loudness(x)) < 0.05
    tp = master.StreamingTruePeak()
    for a in range(0, len(x), 30001):
        tp.update(x[a:a + 30001])
    assert abs(20 * np.log10(tp.finish()) - master.true_peak_dbtp(x)) < 0.01


def test_similarity_flags_an_exact_repeat(tmp_path):
    sr = 48000
    rng = np.random.default_rng(1)
    bar = sr * 2
    bars = [rng.standard_normal(bar) * np.sin(np.linspace(0, 40 * (k + 1), bar)) for k in range(48)]
    bars[40:48] = bars[0:8]  # window 5 repeats window 0
    x = np.concatenate(bars) * 0.1
    f = tmp_path / "rep.wav"
    sf.write(str(f), np.stack([x, x], 1), sr)
    rep = analysis.self_similarity(f, [k * bar for k in range(49)], threshold=0.95)
    assert any({fl["a"], fl["b"]} == {0, 5} for fl in rep["flags"])


def test_neural_layer_skips_unavailable_backend(tmp_path):
    p = planner.plan("techno", 0.05, seed=2, constraints={"neural": {"backend": "no-such-model", "types": ["all"]}}).data
    rec = neural.generate_layer(p, 0, tmp_path)
    assert rec["status"] == "skipped" and rec["reason"]


def test_rf64_and_flac_tagging(tmp_path):
    fields = outputs.tag_fields("t", "techno", "AI-generated test", "0" * 64, {"seed": 1})
    x = np.zeros((4800, 2), np.float32)
    rf = tmp_path / "a.wav"
    with sf.SoundFile(str(rf), "w", 48000, 2, "PCM_24", format="RF64") as f:
        f.write(x)
    outputs.append_ixml(rf, fields)
    assert sf.info(str(rf)).frames == 4800 and "AI_GENERATED=1" in outputs.read_ixml(rf)
    fl = tmp_path / "a.flac"
    sf.write(str(fl), x, 48000, subtype="PCM_24")
    outputs.write_flac_tags(fl, fields)
    from mutagen.flac import FLAC

    assert FLAC(str(fl))["AI_GENERATED"] == ["1"]
    assert outputs.choose_format(5 * 3600)[0] == "FLAC" and outputs.choose_format(3600)[0] == "WAV"
    with pytest.raises(ValueError):
        outputs.choose_format(4 * 3600, "wav")
