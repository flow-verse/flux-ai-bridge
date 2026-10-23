# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a set plan end to end, with per-section checkpointing.

Run directory layout (all of it is regenerable output, never committed):

    plan.json                 the plan that was executed
    midi/<id>.mid             symbolic section MIDI (cheap, regenerated each run)
    neural/<hash>.*           neural layer cache
    sections/<id>-<hash>/     mix.flac, mix.low.flac, meta.json  (checkpoint unit)
    <slug>.wav|.flac          mastered output with AI-disclosure tags
    <slug>.cue, chapters.txt, provenance.json, similarity.json, report.json

A section directory only appears (atomic rename) once its render is complete,
and its name includes a content hash of everything that affects its audio, so
a killed run restarted with the same plan re-renders only what is missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from . import analysis, genres, master, neural, outputs, planner, symbolic, timing, transitions
from .render import BuiltinSynthRenderer, Renderer, get_renderer


class RunInterrupted(RuntimeError):
    pass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "set"


def section_hash(plan: dict, i: int, midi_bytes: bytes, renderer: dict, neural_rec: dict) -> str:
    s = plan["sections"][i]
    g = genres.load(s.get("genre", plan["genre"]))
    ident = {
        "section": {k: v for k, v in s.items() if k not in ("transition_out",)},
        "midi": hashlib.sha256(midi_bytes).hexdigest(),
        "genre": genres.digest(g),
        "renderer": renderer,
        "neural": {k: neural_rec.get(k) for k in ("status", "sha256", "gain_db", "start_bar")} if neural_rec else None,
        "generator": symbolic.GENERATOR_VERSION,
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _render_job(renderer_spec: tuple[str, dict], midi: str, section: dict, out_dir: str,
                neural_rec: dict | None, layer_offset: int) -> dict:
    """Render one section into out_dir atomically. Runs in a worker process."""
    import soundfile as sf
    from scipy import signal

    name, kw = renderer_spec
    r = get_renderer(name, **kw) if isinstance(name, str) else name
    final = Path(out_dir)
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    res = r.render(Path(midi), section, tmp / "mix.flac")
    mix_p, low_p = Path(res.mix), tmp / "mix.low.flac"
    scale = res.scale
    if res.low is None or not Path(res.low).exists() or (neural_rec and neural_rec.get("status") == "ok"):
        mix, sr = sf.read(str(mix_p), dtype="float64", always_2d=True)
        mix /= scale
        if res.low is not None and Path(res.low).exists():
            low = sf.read(str(res.low), dtype="float64", always_2d=True)[0] / scale
        else:  # crossover fallback for renderers without a low stem
            sos = signal.butter(4, 150, "lowpass", fs=sr, output="sos")
            low = signal.sosfiltfilt(sos, mix, axis=0)
        if neural_rec and neural_rec.get("status") == "ok":
            x = sf.read(neural_rec["path"], dtype="float64", always_2d=True)[0]
            sos = signal.butter(2, 120, "highpass", fs=sr, output="sos")
            x = signal.sosfilt(sos, x, axis=0) * 10 ** (neural_rec.get("gain_db", -10) / 20)
            end = min(len(mix), layer_offset + len(x))
            mix[layer_offset:end] += x[: end - layer_offset]
        peak = max(np.max(np.abs(mix)), np.max(np.abs(low)), 1e-9)
        scale = min(1.0, 0.98 / peak)
        sf.write(str(tmp / "mix.flac"), mix * scale, sr, subtype="PCM_24")
        sf.write(str(low_p), low * scale, sr, subtype="PCM_24")
    elif Path(res.low) != low_p:
        os.replace(res.low, low_p)
    meta = {"id": section["id"], "scale": scale, "frames": res.frames, "body_frames": res.body_frames,
            "sample_rate": res.sample_rate, "render_seconds": round(time.time() - t0, 3), "stats": res.stats,
            "neural": (neural_rec or {}).get("status")}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return meta


def _versions() -> dict:
    from importlib import metadata

    try:
        pkg = metadata.version("flux-ai")
    except Exception:
        pkg = "unknown"
    return {"package_version": pkg, "generator": symbolic.GENERATOR_VERSION, "python": sys.version.split()[0],
            "platform": platform.platform()}


def run(plan: planner.SetPlan | dict, out_dir: Path, renderer: Renderer | None = None, workers: int | None = None,
        fmt: str = "auto", progress: Callable[[str, float, str], None] | None = None,
        max_renders: int | None = None, similarity_threshold: float = 0.95) -> dict:
    """Execute a plan into out_dir. Returns the run report (also written to report.json)."""
    t_start = time.time()
    stages: dict[str, float] = {}
    data = planner.normalise(plan.data if isinstance(plan, planner.SetPlan) else plan)
    planner.validate(data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = int(data["sample_rate"])
    renderer = renderer or BuiltinSynthRenderer(sample_rate=sr)
    rid = renderer.identity()
    last = {"frac": -1.0}

    def say(stage: str, frac: float, msg: str = "") -> None:
        # throttle message-less updates to 1 % steps so long runs do not flood the event stream
        if progress is None or (not msg and frac - last["frac"] < 0.01):
            return
        last["frac"] = frac
        progress(stage, frac, msg)
    (out / "plan.json").write_text(json.dumps(data, indent=1) + "\n")
    secs = data["sections"]
    n = len(secs)

    # 1. symbolic MIDI
    t = time.time()
    midi_dir = out / "midi"
    midi_dir.mkdir(exist_ok=True)
    midis = []
    for i, s in enumerate(secs):
        p = midi_dir / f"{s['id']}.mid"
        symbolic.write_midi(data, i, p)
        midis.append(p)
    stages["compose_midi"] = time.time() - t
    say("compose", 0.05, f"{n} sections written as MIDI")

    # 2. optional neural layers (sequential: GPU-bound, cached by request hash)
    t = time.time()
    neural_recs: dict[int, dict] = {}
    for i, s in enumerate(secs):
        if s.get("neural"):
            neural_recs[i] = neural.generate_layer(data, i, out / "neural", sr)
            say("neural", 0.05 + 0.05 * (i + 1) / n, f"{s['id']}: {neural_recs[i].get('status')}")
    stages["neural"] = time.time() - t

    # 3. render missing sections (checkpointed)
    t = time.time()
    sec_root = out / "sections"
    sec_root.mkdir(exist_ok=True)
    todo, dirs = [], []
    for i, s in enumerate(secs):
        h = section_hash(data, i, midis[i].read_bytes(), rid, neural_recs.get(i, {}))
        d = sec_root / f"{s['id']}-{h}"
        dirs.append(d)
        if not (d / "meta.json").exists():
            todo.append(i)
    keep = {d.name for d in dirs}
    for stale in sec_root.iterdir():  # leftovers from killed renders or older plans
        if stale.name.endswith(".tmp"):
            shutil.rmtree(stale, ignore_errors=True)
    cached = n - len(todo)
    say("render", 0.1, f"{cached} sections cached, {len(todo)} to render")
    places = timing.timeline(data)

    def job_args(i):
        off = 0
        rec = neural_recs.get(i)
        if rec and rec.get("status") == "ok":
            off = timing.us_to_samples(places[i].bar_starts_us[rec["start_bar"]], sr)
        spec = (renderer.name, {"sample_rate": renderer.sample_rate, "tail_seconds": renderer.tail_seconds})
        return spec, str(midis[i]), secs[i], str(dirs[i]), rec, off

    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 2))
    rendered = 0
    custom = type(renderer) not in (BuiltinSynthRenderer,)
    if workers <= 1 or custom or len(todo) <= 1:
        for i in todo:
            if max_renders is not None and rendered >= max_renders:
                raise RunInterrupted(f"stopped after {rendered} renders (max_renders)")
            spec, *rest = job_args(i)
            _render_job((renderer, {}) if custom else spec, *rest)
            rendered += 1
            say("render", 0.1 + 0.6 * (cached + rendered) / n, secs[i]["id"])
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_render_job, *job_args(i)): i for i in todo}
            for fut in as_completed(futs):
                fut.result()
                rendered += 1
                say("render", 0.1 + 0.6 * (cached + rendered) / n, secs[futs[fut]]["id"])
    stages["render"] = time.time() - t
    for stale in sec_root.iterdir():
        if stale.is_dir() and stale.name not in keep:
            shutil.rmtree(stale, ignore_errors=True)

    # 4. assemble transitions + master (streaming)
    t = time.time()
    audio = []
    for i, d in enumerate(dirs):
        meta = json.loads((d / "meta.json").read_text())
        audio.append(transitions.SectionAudio(i, d / "mix.flac", d / "mix.low.flac", meta["scale"], meta["frames"]))
    tl = transitions.Timeline(data, audio, sr)
    total_seconds = tl.total / sr
    sf_format, subtype, ext = outputs.choose_format(total_seconds, fmt)
    slug = _slug(data.get("title", data["genre"]))
    master_path = out / f"{slug}{ext}"
    tmp_master = out / f".{slug}.partial{ext}"
    neural_used = [r for r in neural_recs.values() if r.get("status") == "ok"]
    disclosure = outputs.disclosure_text(neural_used, data.get("human_decisions", []))
    title = data.get("title", data["genre"])

    import soundfile as sf

    def writer():
        f = sf.SoundFile(str(tmp_master), "w", sr, 2, subtype, format=sf_format)
        outputs.set_info_strings(f, title, data["genre"], "AI_GENERATED=1; " + disclosure)
        return f

    tgt = data.get("target", {})
    mres = master.master(tl.read, tl.total, sr, writer, target_lufs=float(tgt.get("lufs", -14.0)),
                         tolerance=float(tgt.get("lufs_tolerance", 0.5)), tp_dbtp=float(tgt.get("true_peak_dbtp", -1.0)),
                         progress=lambda f: say("master", 0.7 + 0.2 * f, ""))
    joins = tl.joins()
    tl.close()
    stages["master"] = time.time() - t

    # 5. checks on the written master: clicks at joins, self-similarity
    t = time.time()
    clicks = []
    with sf.SoundFile(str(tmp_master)) as f:
        for j in joins:
            for pos in {j["start"], j["end_of_prev_body"], j["end_of_prev_file"], j["swap"] or j["start"]}:
                # read 150 ms of context each side so the detector's 50 ms reference
                # windows are complete; count flags within +-50 ms of the join point
                a = max(0, pos - int(0.15 * sr))
                f.seek(a)
                x = f.read(int(0.3 * sr), dtype="float64", always_2d=True)
                for k in analysis.find_discontinuities(x, sr):
                    if abs(a + k - pos) <= int(0.05 * sr):
                        clicks.append({"join": f"{j['from']}->{j['to']}", "sample": int(a + k)})
    sim = analysis.self_similarity(tmp_master, tl.bar_starts(), threshold=similarity_threshold)
    (out / "similarity.json").write_text(json.dumps(sim, indent=1))
    stages["analysis"] = time.time() - t
    say("analysis", 0.95, f"similarity max {sim.get('max_nonadjacent')}")

    # 6. metadata, provenance, disclosure tags
    chapters = outputs.chapter_points(data, tl)
    (out / "chapters.txt").write_text(outputs.chapters_txt(chapters, total_seconds))
    (out / f"{slug}.cue").write_text(outputs.cue_sheet(chapters, master_path.name, title, data["genre"], disclosure))
    gtabs = {}
    for s in secs:
        g = genres.load(s.get("genre", data["genre"]))
        gtabs[g["name"]] = {"file": g["_file"], "sha256": g["_sha256"], "status": "starting values for owner curation"}
    master_info = {"target_lufs": tgt.get("lufs", -14.0), "target_true_peak_dbtp": tgt.get("true_peak_dbtp", -1.0),
                   **mres.__dict__, "format": sf_format, "subtype": subtype, "sample_rate": sr}
    os.replace(tmp_master, master_path)
    prov = outputs.provenance(data, rid, list(neural_recs.values()), master_info,
                              {"master": master_path.name, "cue": f"{slug}.cue", "chapters": "chapters.txt"},
                              disclosure, gtabs, _versions())
    prov_text = json.dumps(prov, indent=1, default=str)
    (out / "provenance.json").write_text(prov_text)
    fields = outputs.tag_fields(title, data["genre"], disclosure, hashlib.sha256(prov_text.encode()).hexdigest(), data)
    if sf_format == "FLAC":
        outputs.write_flac_tags(master_path, fields)
    else:
        outputs.append_ixml(master_path, fields)

    wall = time.time() - t_start
    sizes = {p.name: p.stat().st_size for p in [master_path, out / "provenance.json", out / f"{slug}.cue"]}
    sizes["section_cache_bytes"] = sum(f.stat().st_size for f in sec_root.rglob("*") if f.is_file())
    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    div = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux KiB
    report = {
        "title": title, "genre": data["genre"], "seed": data["seed"], "sections": n,
        "segments": len({s["segment"] for s in secs}),
        "audio_seconds": round(total_seconds, 3), "audio_hms": outputs._hms(total_seconds, True),
        "wall_clock_seconds": round(wall, 2), "speed_factor": round(total_seconds / wall, 2),
        "stages_seconds": {k: round(v, 2) for k, v in stages.items()},
        "sections_rendered": rendered, "sections_cached": cached, "workers": workers, "renderer": rid,
        "loudness_lufs": mres.lufs, "true_peak_dbtp": mres.true_peak_dbtp, "premaster_lufs": mres.premaster_lufs,
        "gain_db": mres.gain_db, "limiter_max_reduction_db": mres.limiter_max_reduction_db, "master_passes": mres.passes,
        "loudness_in_spec": bool(abs(mres.lufs - float(tgt.get("lufs", -14.0))) <= float(tgt.get("lufs_tolerance", 0.5))),
        "true_peak_in_spec": bool(mres.true_peak_dbtp <= float(tgt.get("true_peak_dbtp", -1.0))),
        "join_clicks": len(clicks), "join_click_samples": clicks[:20], "joins_checked": len(joins),
        "similarity": {k: sim.get(k) for k in ("windows", "threshold", "max_nonadjacent", "flag_count", "percentiles")},
        "neural_layers": {"requested": len(neural_recs), "used": len(neural_used),
                          "skipped": [r.get("reason") for r in neural_recs.values() if r.get("status") == "skipped"]},
        "no_repeat_violations": planner.no_repeat_violations(data),
        "file_sizes_bytes": sizes,
        "peak_rss_mb": {"main": round(ru_self / div / 2**20, 1), "largest_worker": round(ru_kids / div / 2**20, 1)},
        "outputs": {"master": str(master_path), "cue": str(out / f"{slug}.cue"), "chapters": str(out / "chapters.txt"),
                    "provenance": str(out / "provenance.json"), "similarity": str(out / "similarity.json"),
                    "plan": str(out / "plan.json"), "report": str(out / "report.json")},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    say("done", 1.0, str(master_path))
    return report
