# SPDX-License-Identifier: GPL-3.0-or-later
"""Output files: master writer, cue sheet, chapters, provenance and AI-disclosure tags."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import struct
from pathlib import Path
from xml.sax.saxutils import escape

from . import theory

FOUR_HOURS = 4 * 3600
TOOL = "flux-ai longform"
DISCLOSURE = ("AI-generated music. Composed by the Flux long-form orchestrator: an algorithmic planner and "
              "seeded symbolic generator, rendered by synthesizers and mastered automatically{neural}. "
              "Human input: genre, duration, seed{human}. See provenance.json.")


def choose_format(seconds: float, fmt: str = "auto") -> tuple[str, str, str]:
    """(soundfile format, subtype, extension). WAV below 4 h; FLAC (default) or RF64 from 4 h."""
    fmt = fmt.lower()
    if fmt == "auto":
        fmt = "wav" if seconds < FOUR_HOURS else "flac"
    if fmt == "wav":
        if seconds >= FOUR_HOURS:
            raise ValueError("24-bit stereo 48 kHz WAV cannot hold 4 h (RIFF 4 GiB limit); use flac or rf64")
        return "WAV", "PCM_24", ".wav"
    if fmt == "rf64":
        return "RF64", "PCM_24", ".wav"
    if fmt == "flac":
        return "FLAC", "PCM_24", ".flac"
    raise ValueError(f"unknown format {fmt!r}")


def disclosure_text(neural_used: list[dict], human_decisions: list) -> str:
    neural = ""
    if neural_used:
        names = sorted({f"{r.get('result_backend', r['backend'])}/{r.get('model', '?')}" for r in neural_used})
        neural = "; with neural audio layers from " + ", ".join(names)
    human = f"; {len(human_decisions)} logged human decisions" if human_decisions else ""
    return DISCLOSURE.format(neural=neural, human=human)


def set_info_strings(f, title: str, genre: str, comment: str) -> None:
    """libsndfile strings: WAV/RF64 LIST-INFO chunk, FLAC Vorbis comments."""
    f.title = title
    f.artist = "Flux long-form generator (AI-generated)"
    f.genre = genre
    f.comment = comment
    f.software = TOOL
    f.date = _dt.date.today().isoformat()


def ixml_chunk(fields: dict) -> bytes:
    user = "\n".join(f"{k}={v}" for k, v in fields.items())
    body = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<BWFXML><IXML_VERSION>2.10</IXML_VERSION>"
            f"<PROJECT>{escape(str(fields.get('TITLE', '')))}</PROJECT>"
            f"<NOTE>{escape(str(fields.get('AI_DISCLOSURE', '')))}</NOTE>"
            f"<USER>{escape(user)}</USER></BWFXML>").encode("utf-8")
    if len(body) % 2:
        body += b"\0"
    return b"iXML" + struct.pack("<I", len(body)) + body


def append_ixml(path: Path, fields: dict) -> None:
    """Append an iXML chunk to a finished WAV or RF64 file and fix the RIFF size."""
    chunk = ixml_chunk(fields)
    with open(path, "r+b") as f:
        head = f.read(12)
        f.seek(0, 2)
        f.write(chunk)
        size = f.tell()
        if head[:4] == b"RIFF":
            f.seek(4)
            f.write(struct.pack("<I", size - 8))
        elif head[:4] == b"RF64":
            f.seek(12)
            if f.read(4) != b"ds64":
                raise ValueError("RF64 without ds64 chunk")
            f.seek(20)
            f.write(struct.pack("<Q", size - 8))
        else:
            raise ValueError("not a RIFF/RF64 file")


def read_ixml(path: Path) -> str | None:
    with open(path, "rb") as f:
        head = f.read(12)
        ds64_data = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"ds64" and head[:4] == b"RF64":
                ds64_data = struct.unpack("<Q", f.read(16)[8:16])[0]
                f.seek(size - 16, 1)
                continue
            if cid == b"data" and size == 0xFFFFFFFF and ds64_data is not None:
                size = ds64_data
            if cid == b"iXML":
                return f.read(size).decode("utf-8").rstrip("\0")
            f.seek(size + (size & 1), 1)


def tag_fields(title: str, genre: str, disclosure: str, provenance_sha: str, plan: dict) -> dict:
    return {
        "TITLE": title,
        "GENRE": genre,
        "AI_GENERATED": "1",
        "AI_DISCLOSURE": disclosure,
        "AI_SYSTEM": TOOL,
        "AI_PLAN_SEED": str(plan.get("seed")),
        "PROVENANCE_SHA256": provenance_sha,
        "DDEX_AI_STYLE": "composition=algorithmic;instrumentation=synthesized;vocals=none;post_production=automated",
    }


def write_flac_tags(path: Path, fields: dict) -> None:
    from mutagen.flac import FLAC

    f = FLAC(str(path))
    for k, v in fields.items():
        f[k] = str(v)
    f["COMMENT"] = fields["AI_DISCLOSURE"]
    f["ARTIST"] = "Flux long-form generator (AI-generated)"
    f.save()


# ------------------------------------------------------------------ chapters / cue

def chapter_points(plan: dict, timeline) -> list[dict]:
    """One chapter per segment, at the incoming bass swap (or start) of its first section."""
    out = []
    secs = plan["sections"]
    sr = timeline.sr
    seen = set()
    for pl in timeline.placed:
        s = secs[pl.index]
        if s["segment"] in seen:
            continue
        seen.add(s["segment"])
        if pl.index == 0:
            t = 0
        elif pl.swap_at is not None:
            t = pl.swap_at
        elif pl.ov_in:
            t = pl.start + pl.ov_in // 2
        else:
            t = pl.start
        key = s["key"]
        cam = s.get("camelot")
        key_txt = f"{theory.key_str(key)}" + (f" ({cam})" if cam else "")
        out.append({"segment": s["segment"], "seconds": t / sr,
                    "title": f"Part {s['segment'] + 1} - {key_txt} - {s['bpm']:g} BPM"})
    return out


def _hms(sec: float, hours: bool) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if hours else f"{m}:{s:02d}"


def chapters_txt(chapters: list[dict], total_seconds: float) -> str:
    hours = total_seconds >= 3600
    return "".join(f"{_hms(c['seconds'], hours)} {c['title']}\n" for c in chapters)


def cue_sheet(chapters: list[dict], audio_name: str, title: str, genre: str, disclosure: str) -> str:
    ftype = "WAVE" if audio_name.lower().endswith(".wav") else "FLAC"
    lines = [f'REM GENRE "{genre}"', f'REM COMMENT "{disclosure}"', "REM AI_GENERATED 1",
             'PERFORMER "Flux long-form generator (AI-generated)"', f'TITLE "{title}"', f'FILE "{audio_name}" {ftype}']
    for n, c in enumerate(chapters, 1):
        frames = int(round(c["seconds"] * 75))
        mm, rem = divmod(frames, 75 * 60)
        ss, ff = divmod(rem, 75)
        lines += [f"  TRACK {n:02d} AUDIO", f'    TITLE "{c["title"]}"',
                  '    PERFORMER "Flux long-form generator (AI-generated)"', f"    INDEX 01 {mm:02d}:{ss:02d}:{ff:02d}"]
    return "\n".join(lines) + "\n"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


DECISION_LOG_NOTE = (
    "Ordered log of decisions that shaped this audio. 'human' entries come from the plan's human_decisions "
    "(genre, duration, seed, plan edits recorded by the caller). 'agent' entries are the flux-engine agent "
    "turns the flux-host renderer made while building each section's project (labels from end_agent_turn, "
    "plus the engine's get_history summary); they are automated, not human. 'human_review' is empty until "
    "a human-review step records who listened, when, the verdict per section and any edits (as further "
    "engine turns or plan changes).")


def decision_log(plan: dict, metas: list[dict], section_dirs: list) -> dict:
    """The human-decision log: human plan decisions, then per-section engine agent turns (flux-host)."""
    entries = [{"actor": "human", "source": "plan.human_decisions", "decision": d}
               for d in plan.get("human_decisions", [])]
    history = {}
    for meta, d in zip(metas, section_dirs):
        eng = (meta.get("stats") or {}).get("engine")
        if not eng:
            continue
        sid = meta["id"]
        history[sid] = {"get_history": eng.get("history"), "project": f"{d}/{eng.get('project', 'project')}"}
        for t in eng.get("turns", []):
            entries.append({"actor": "agent", "source": "flux-engine", "section": sid, "turn": t.get("label"),
                            "recorded": t.get("recorded")})
    return {"schema": "flux.decision_log/0.1", "note": DECISION_LOG_NOTE, "entries": entries,
            "engine_history": history, "human_review": None}


def section_rendering(metas: list[dict]) -> dict:
    """Per-section instruments and engine version, for renderers that report them (flux-host)."""
    out = {}
    for m in metas:
        st = m.get("stats") or {}
        if "plugins" in st:
            out[m["id"]] = {"plugins": st["plugins"], "engine_api_version": (st.get("engine") or {}).get("api_version"),
                            "conversion": st.get("conversion")}
    return out


def provenance(plan: dict, renderer: dict, neural_records: list[dict], master: dict, outputs: dict,
               disclosure: str, genre_tables: dict, versions: dict, decision_log: dict | None = None,
               section_rendering: dict | None = None) -> dict:
    neural_used = [r for r in neural_records if r.get("status") == "ok"]
    return {
        "schema": "flux.provenance/0.1",
        "ai_generated": True,
        "disclosure": disclosure,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": {"name": TOOL, **versions},
        "ddex_style_ai_disclosure": {
            "note": "DDEX-style fields; exact DDEX field names still need verifying against DDEX primary sources (P6).",
            "composition": "AI/algorithmic (seeded planner + symbolic generator)",
            "instrumentation": ("synthesized" if renderer.get("name") != "flux-host" else
                                "plugin instruments in flux-host (" + ", ".join(sorted({p.get("name", "?") for p in
                                (renderer.get("plugins") or {}).values()})) + ")"
                                + ("" if renderer.get("drums") == "plugin" else " + built-in synthesized drums"))
                               + (" + neural audio layers" if neural_used else ""),
            "vocals": "none",
            "post_production": "automated loudness normalisation and true-peak limiting",
        },
        "components": [
            {"component": "planner", "name": "flux_ai.longform.planner", "licence": "GPL-3.0-or-later"},
            {"component": "symbolic generator", "name": "flux_ai.longform.symbolic", "licence": "GPL-3.0-or-later",
             "version": versions.get("generator")},
            {"component": "renderer", **renderer, "licence": "GPL-3.0-or-later"},
        ],
        "genre_tables": genre_tables,
        "seeds": {"plan": plan.get("seed"),
                  "sections": {s["id"]: s["seeds"] for s in plan["sections"]}},
        "neural_layers": neural_records,
        "models": sorted({(r.get("result_backend", r["backend"]), r.get("model", "?"),
                           (r.get("provenance") or {}).get("licence", "unknown")) for r in neural_used}),
        "human_decisions": plan.get("human_decisions", []),
        "decision_log": decision_log or {"schema": "flux.decision_log/0.1", "note": DECISION_LOG_NOTE,
                                         "entries": [], "engine_history": {}, "human_review": None},
        "section_rendering": section_rendering or {},
        "mastering": master,
        "outputs": outputs,
        "plan": plan,
        "watermark": None,
    }
