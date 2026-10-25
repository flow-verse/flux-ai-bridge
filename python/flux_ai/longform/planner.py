# SPDX-License-Identifier: GPL-3.0-or-later
"""Set planner: genre + duration + seed -> a set plan (set -> section -> bar).

The plan is plain JSON (see schema/set_plan.schema.json) so an LLM agent can
write or edit it and pass it back with `flux-ai longform --plan plan.json`.
Everything here is deterministic for a given (genre table, hours, seed,
constraints).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

from . import genres, theory, timing

SCHEMA_PATH = Path(__file__).with_name("schema") / "set_plan.schema.json"
PATTERN_FIELDS = ("form", "drums", "bass", "progression", "chord_rhythm", "motif")
ALL_LAYERS = ["kick", "clap", "hat_closed", "hat_open", "perc", "ride", "bass", "chords", "pad", "lead"]


def derive_seed(*parts) -> int:
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


class PlanError(ValueError):
    pass


class SetPlan:
    """Thin wrapper around the plan dict (the dict is the source of truth)."""

    def __init__(self, data: dict):
        self.data = data

    @property
    def sections(self) -> list[dict]:
        return self.data["sections"]

    def to_json(self, path: Path | None = None) -> str:
        text = json.dumps(self.data, indent=1, sort_keys=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def validate(self) -> None:
        validate(self.data)

    def duration_seconds(self) -> float:
        return timing.total_us(self.data) / 1e6

    @classmethod
    def from_file(cls, path: Path) -> "SetPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls(normalise(data))
        p.validate()
        return p


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(data: dict) -> None:
    import jsonschema

    try:
        jsonschema.validate(data, schema())
    except jsonschema.ValidationError as e:
        loc = "/".join(str(p) for p in e.absolute_path)
        raise PlanError(f"plan invalid at {loc or '<root>'}: {e.message}") from None
    ids = [s["id"] for s in data["sections"]]
    if len(set(ids)) != len(ids):
        raise PlanError("section ids must be unique")
    motif_ids = {m["id"] for m in data.get("motifs", [])}
    for s in data["sections"]:
        g = genres.load(s.get("genre", data["genre"]))
        theory.scale_cents(s["key"], g.get("maqamat"))  # raises on unknown mode
        m = s.get("motif")
        if m and m.get("id") not in motif_ids:
            raise PlanError(f"section {s['id']} references unknown motif {m.get('id')!r}")


def normalise(data: dict) -> dict:
    """Fill fields an externally written plan may omit (layers, bar_plan, camelot...)."""
    data = copy.deepcopy(data)
    data.setdefault("schema_version", "1.0")
    data.setdefault("kind", "flux.longform.set_plan")
    data.setdefault("seed", 0)
    data.setdefault("sample_rate", 48000)
    data.setdefault("target", {"lufs": -14.0, "lufs_tolerance": 0.5, "true_peak_dbtp": -1.0})
    data.setdefault("human_decisions", [])
    for i, s in enumerate(data["sections"]):
        g = genres.load(s.get("genre", data["genre"]))
        s.setdefault("id", f"s{i:03d}")
        s.setdefault("segment", i)
        s.setdefault("seeds", {"section": derive_seed(data["seed"], i, "section")})
        s.setdefault("transition_out", {"type": "continue", "bars": 0})
        s.setdefault("energy_end", s["energy"])
        s.setdefault("camelot", theory.camelot(s["key"]))
        s.setdefault("layers", _layers(g, s["type"], s["energy"]))
        s.setdefault("patterns", {})
        pats = s["patterns"]
        pats.setdefault("drums", sorted(g["drums"]["patterns"])[0])
        pats.setdefault("bass", sorted(g["bass"]["patterns"])[0])
        pats.setdefault("progression", sorted(g["chords"]["progressions"])[0])
        pats.setdefault("chord_rhythm", sorted(g["chords"]["rhythms"])[0])
        s.setdefault("motif", None)
        s.setdefault("neural", None)
        if "bar_plan" not in s:
            s["bar_plan"] = _bar_plan(g, s, random.Random(s["seeds"]["section"]))
    return data


# ---------------------------------------------------------------- helpers

def _curve_at(points: list, x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def _weighted(rng: random.Random, items: list, weights: list):
    total = sum(weights)
    if total <= 0:
        return items[0]
    r = rng.uniform(0, total)
    for it, w in zip(items, weights):
        r -= w
        if r <= 0:
            return it
    return items[-1]


def _pick_no_repeat(rng: random.Random, vocab: list, history: list, window: int):
    """Uniform pick among values not used in the last `window` picks; LRU fallback."""
    vocab = sorted(vocab)
    window = min(window, len(vocab) - 1)
    recent = history[-window:] if window > 0 else []
    cands = [v for v in vocab if v not in recent]
    if not cands:
        last_use = {v: max((i for i, h in enumerate(history) if h == v), default=-1) for v in vocab}
        cands = [min(vocab, key=lambda v: last_use[v])]
    return rng.choice(cands)


def effective_windows(g: dict) -> dict:
    vocab = {
        "form": len(g["segment"]["forms"]),
        "drums": len(g["drums"]["patterns"]),
        "bass": len(g["bass"]["patterns"]),
        "progression": len(g["chords"]["progressions"]),
        "chord_rhythm": len(g["chords"]["rhythms"]),
        "motif": int(g.get("motif_pool", 8)),
    }
    nr = g.get("no_repeat", {})
    return {k: max(0, min(int(nr.get(k, 1)), vocab[k] - 1)) for k in PATTERN_FIELDS}


def _layers(g: dict, sec_type: str, energy: float, keep: tuple = ()) -> list[str]:
    """Layers of a section: those over their energy threshold minus the type's mutes plus its adds.
    `keep` exempts layers from the mute list (the set's final outro keeps its harmony)."""
    st = g["section_types"].get(sec_type, {})
    thr = g["layers"]
    on = [l for l in ALL_LAYERS if energy >= float(thr.get(l, 1.1))]
    on = [l for l in on if l not in st.get("mute", []) or l in keep]
    for l in st.get("add", []):
        if l not in on:
            on.append(l)
    return [l for l in ALL_LAYERS if l in on]


def _bar_plan(g: dict, s: dict, rng: random.Random) -> list[dict]:
    prog = g["chords"]["progressions"][s["patterns"]["progression"]]
    e0, e1 = s["energy"], s.get("energy_end", s["energy"])
    out = []
    no_fill = s["type"] in ("intro", "outro", "breakdown")
    for b in range(s["bars"]):
        x = b / max(1, s["bars"] - 1)
        out.append({
            "chord": int(prog[b % len(prog)]),
            "energy": round(e0 + (e1 - e0) * x, 3),
            "phrase": b // 8,
            "variation": rng.randrange(1000),
            "fill": ((not no_fill) and (b % 8 == 7) and (b // 8) % 2 == 1) or (s["type"] == "build" and b == s["bars"] - 1),
        })
    return out


def _make_motifs(g: dict, rng: random.Random) -> list[dict]:
    lead = g["lead"]
    lo, hi = lead.get("range_degrees", [-2, 9])
    stepwise = bool(lead.get("stepwise", False))
    # optional melodic style: per-degree preference (e.g. the jins of a maqam) and the degrees a
    # phrase may rest on (its last note snaps to the nearest one)
    pref = {int(k): float(v) for k, v in (lead.get("degree_weights") or {}).items()}
    rest = [int(x) for x in (lead.get("rest_degrees") or [])]
    step_w = [2, 4, 4, 2, 1, 1, 0.5 if not stepwise else 0, 0.5 if not stepwise else 0]
    if stepwise:
        step_w[0] = step_w[3] = 1.0   # +-2 (a third) rarer, +-3 rare: conjunct lines
        step_w[4] = step_w[5] = 0.3
    motifs = []
    for k in range(int(g.get("motif_pool", 8))):
        bars = rng.choice(lead.get("motif_bars", [1, 2]))
        n_lo, n_hi = lead.get("notes_per_bar", [3, 6])
        n = max(1, sum(rng.randint(n_lo, n_hi) for _ in range(bars)))
        steps = 16 * bars
        # bias onsets toward eighth notes, keep them unique and sorted
        pool = [s for s in range(steps) if s % 2 == 0] * 2 + list(range(steps))
        starts = sorted(set(rng.sample(pool, min(len(pool), n * 2))))[:n]
        if starts[0] != 0 and rng.random() < 0.6:
            starts[0] = 0
        starts = sorted(set(starts))
        notes, deg = [], rng.choice(list(lead.get("start_degrees", [0, 2, 4])))
        for j, st in enumerate(starts):
            nxt = starts[j + 1] if j + 1 < len(starts) else steps
            length = max(1, min(nxt - st, rng.choice([1, 2, 2, 3, 4])))
            vel = round(rng.uniform(0.62, 1.0), 2)
            notes.append([st, length, deg, vel])
            cands = [-2, -1, 1, 2, -3, 3, 4, -4]
            w = [sw * pref.get(min(hi, max(lo, deg + st)), 1.0) for st, sw in zip(cands, step_w)]
            if stepwise:  # below the tonic only as a lower neighbour: down one step to it, up one step back
                w = [0.0 if ((deg + st < 0 and st != -1) or (deg < 0 and st != 1)) else x for st, x in zip(cands, w)]
            step = _weighted(rng, cands, w)
            deg = min(hi, max(lo, deg + step))
        if rest and notes:
            last = notes[-1][2]
            r = min(rest, key=lambda d: (abs(d - last), d))
            notes[-1][2] = r
            if len(notes) >= 3:  # cadence: approach the resting note by step from the side the line came from
                notes[-2][2] = r + 1 if notes[-3][2] > r else r - 1
        motifs.append({"id": f"m{k:02d}", "bars": bars, "notes": notes})
    return motifs


def _next_key(g: dict, rng: random.Random, key: dict, recent: list[str]) -> tuple[dict, str]:
    if g.get("key_system") == "maqam":
        moves = g["modulations"][key["mode"]]
        mv = _weighted(rng, moves, [m.get("weight", 1) for m in moves])
        pc = (theory.pitch_class(key["tonic"]) + int(mv["shift"])) % 12
        return {"tonic": theory.note_name(pc), "mode": mv["to"]}, f"{key['mode']}->{mv['to']} ({mv['shift']:+d} st)"
    ks = g["keys"]
    code = theory.camelot(key)
    moves = theory.camelot_neighbours(code)
    weights = []
    for name, target in moves:
        w = float(ks.get("moves", {}).get(name, 1))
        if name != "same" and target in recent:
            w *= 0.25  # discourage bouncing back to a key just used
        if name == "same" and recent[-2:] == [code, code]:
            w = 0.0  # never three segments in a row on one key
        weights.append(w)
    name, target = _weighted(rng, moves, weights)
    return theory.from_camelot(target, ks.get("minor_mode", "minor"), ks.get("major_mode", "major")), name


def _parse_key(text: str) -> dict:
    tonic, mode = text.split(None, 1)
    return {"tonic": tonic, "mode": mode.strip().lower()}


# ---------------------------------------------------------------- planner

def plan(genre: str, hours: float, seed: int = 0, constraints: dict | None = None) -> SetPlan:
    if hours <= 0:
        raise PlanError("hours must be > 0")
    c = dict(constraints or {})
    g = genres.load(genre)
    rng = random.Random(derive_seed(seed, "plan", g["name"]))
    total_s = hours * 3600.0

    curves = g["energy_curves"]
    ec = c.get("energy_curve")
    if isinstance(ec, list):
        curve_name, points = "custom", ec
    else:
        curve_name = ec if isinstance(ec, str) else rng.choice(sorted(curves))
        points = curves[curve_name]

    bpm_lo, bpm_hi = float(g["bpm"]["min"]), float(g["bpm"]["max"])
    if isinstance(c.get("bpm"), (int, float)):
        bpm_lo = bpm_hi = float(c["bpm"])
    elif isinstance(c.get("bpm"), list):
        bpm_lo, bpm_hi = map(float, c["bpm"])
    step_max = float(g.get("bpm_step_max", 2.0))
    seg_minutes = float(c.get("segment_minutes", g["segment"]["target_minutes"]))
    trans = dict(g["segment"]["transition"])
    if "transition_bars" in c:
        trans["bars"] = int(c["transition_bars"])
    windows = effective_windows(g)
    windows.update({k: v for k, v in c.get("no_repeat", {}).items() if k in windows})

    motifs = _make_motifs(g, random.Random(derive_seed(seed, "motifs", g["name"])))
    history: dict[str, list] = {k: [] for k in PATTERN_FIELDS}
    b_history: list[str] = []
    key_hist: list[str] = []
    key = _parse_key(c["start_key"]) if c.get("start_key") else _parse_key(rng.choice(g["keys"]["start"]))
    key_move = "start"
    bpm_prev = None
    sections: list[dict] = []
    t = 0.0
    seg = 0
    neural = c.get("neural")  # {"backend": ..., "types": [...], "gain_db": ...}

    while t < total_s - 1.0:
        # choose this segment's material (no-repeat windows across segments)
        pick = {}
        pick["form"] = _pick_no_repeat(rng, list(g["segment"]["forms"]), history["form"], windows["form"])
        pick["drums"] = _pick_no_repeat(rng, list(g["drums"]["patterns"]), history["drums"], windows["drums"])
        pick["bass"] = _pick_no_repeat(rng, list(g["bass"]["patterns"]), history["bass"], windows["bass"])
        pick["progression"] = _pick_no_repeat(rng, list(g["chords"]["progressions"]), history["progression"], windows["progression"])
        pick["chord_rhythm"] = _pick_no_repeat(rng, list(g["chords"]["rhythms"]), history["chord_rhythm"], windows["chord_rhythm"])
        pick["motif"] = _pick_no_repeat(rng, [m["id"] for m in motifs], history["motif"], windows["motif"])
        transform = rng.choice(g.get("transforms", ["identity"]))
        for k, v in pick.items():
            history[k].append(v)
        # a second theme for the grooves and breakdowns of this segment, so one segment is not one motif
        # looped; it avoids the A themes in the no-repeat window (which already holds this segment's) and
        # the last two B themes, and is kept out of history["motif"] so the A-theme window stays as declared
        recent = history["motif"][-max(1, windows["motif"]):] + b_history[-2:]
        motif_b = _pick_no_repeat(rng, [m["id"] for m in motifs], recent, len(recent))
        b_history.append(motif_b)
        transform_b = rng.choice(g.get("transforms", ["identity"]))

        form = [list(x) for x in g["segment"]["forms"][pick["form"]]]
        # scale the form to the requested segment length (bars in multiples of 4)
        est_e = _curve_at(points, (t + seg_minutes * 30) / total_s)
        bpm_target = bpm_lo + (bpm_hi - bpm_lo) * min(1.0, max(0.0, 0.15 + 0.8 * est_e)) + rng.uniform(-0.6, 0.6)
        if bpm_prev is not None:
            bpm_target = min(bpm_prev + step_max, max(bpm_prev - step_max, bpm_target))
        bpm = round(min(bpm_hi, max(bpm_lo, bpm_target)) * 2) / 2
        bar_s = timing.BEATS_PER_BAR * 60.0 / bpm
        form_bars = sum(b for _, b in form)
        want_bars = seg_minutes * 60.0 / bar_s
        scale = want_bars / form_bars
        remaining = total_s - t
        seg_s = form_bars * scale * bar_s
        last = remaining - (seg_s - trans["bars"] * bar_s) < 0.6 * seg_s
        if last:
            scale = remaining / (form_bars * bar_s)
        min_bars = max(4, int(trans["bars"]) if not last else 4)
        bars_list = [max(4, int(round(b * scale / 4)) * 4) for _, b in form]
        # transitions need room at both ends of a segment
        bars_list[0] = max(bars_list[0], min_bars if seg > 0 else 4)
        bars_list[-1] = max(bars_list[-1], min_bars if not last else 4)

        seg_start_t = t
        seg_len = sum(bars_list) * bar_s
        for j, ((stype, _), bars) in enumerate(zip(form, bars_list)):
            pos0 = (t - seg_start_t) / seg_len
            sec_mid = seg_start_t + seg_len * (pos0 + bars * bar_s / seg_len / 2)
            stype_cfg = g["section_types"].get(stype, {})
            e = _curve_at(points, sec_mid / total_s) + float(stype_cfg.get("energy", 0.0))
            e_end = _curve_at(points, (t + bars * bar_s) / total_s) + float(stype_cfg.get("energy", 0.0))
            e, e_end = round(min(1.0, max(0.05, e)), 3), round(min(1.0, max(0.05, e_end)), 3)
            if stype == "build":
                e_end = round(min(1.0, e + 0.15), 3)
            idx = len(sections)
            sseed = derive_seed(seed, seg, j, "section")
            s = {
                "id": f"s{idx:03d}",
                "segment": seg,
                "type": stype,
                "genre": g["name"],
                "bpm": bpm,
                "key": dict(key),
                "camelot": theory.camelot(key),
                "key_move": key_move if j == 0 else "same",
                "energy": e,
                "energy_end": e_end,
                "bars": bars,
                "layers": _layers(g, stype, e, keep=("chords", "pad") if last and stype == "outro" else ()),
                "seeds": {"section": sseed, "drums": derive_seed(sseed, "drums"), "bass": derive_seed(sseed, "bass"),
                          "chords": derive_seed(sseed, "chords"), "lead": derive_seed(sseed, "lead"),
                          "render": derive_seed(sseed, "render")},
                "patterns": {"form": pick["form"], "drums": pick["drums"], "bass": pick["bass"],
                             "progression": pick["progression"], "chord_rhythm": pick["chord_rhythm"]},
                "motif": {"id": motif_b, "transform": transform_b} if stype in ("groove", "breakdown")
                         else {"id": pick["motif"], "transform": transform},
                "transition_out": {"type": "continue", "bars": 0},
                "neural": None,
            }
            if neural and (stype in neural.get("types", ["breakdown"]) or "all" in neural.get("types", [])):
                s["neural"] = {
                    "backend": neural["backend"],
                    "prompt": g.get("neural", {}).get("prompt", "{key} {bpm} BPM").format(bpm=bpm, key=theory.key_str(key)),
                    "gain_db": float(neural.get("gain_db", -10.0)),
                    "mode": "generate",
                    "seed": derive_seed(sseed, "neural"),
                }
            s["bar_plan"] = _bar_plan(g, s, random.Random(sseed))
            sections.append(s)
            t += bars * bar_s

        if not last:
            tb = min(int(trans["bars"]), sections[-1]["bars"])
            sections[-1]["transition_out"] = {"type": trans["type"], "bars": tb,
                                              "eq": bool(trans.get("eq", True)), "bass_swap": bool(trans.get("bass_swap", True)),
                                              "swap_at": float(trans.get("swap_at", 0.5))}
            t -= tb * bar_s  # the next segment starts inside this one's last bars
        key_hist.append(theory.camelot(key) or key_str_safe(key))
        key, key_move = _next_key(g, rng, key, key_hist)
        bpm_prev = bpm
        seg += 1
        if last:
            break

    # the next segment's first section must be long enough for the incoming overlap
    for i in range(len(sections) - 1):
        tr = sections[i]["transition_out"]
        if tr["type"] != "continue":
            tr["bars"] = min(tr["bars"], sections[i + 1]["bars"], sections[i]["bars"])

    data = {
        "schema_version": "1.0",
        "kind": "flux.longform.set_plan",
        "title": c.get("title", f"{hours:g} h of {g['name']}"),
        "genre": g["name"],
        "genre_table": {"file": g["_file"], "sha256": g["_sha256"]},
        "seed": int(seed),
        "hours": float(hours),
        "sample_rate": 48000,
        "target": {"lufs": -14.0, "lufs_tolerance": 0.5, "true_peak_dbtp": -1.0},
        "energy_curve": {"name": curve_name, "points": points},
        "constraints": c,
        "motifs": motifs,
        "anti_repetition": {"no_repeat": windows, "unit": "segment"},
        "human_decisions": [],
        "sections": sections,
    }
    p = SetPlan(data)
    p.validate()
    return p


def key_str_safe(key: dict) -> str:
    return theory.key_str(key)


# ---------------------------------------------------------------- checks

def no_repeat_violations(data: dict) -> list[str]:
    """Pattern choices that recur inside their no-repeat window (unit = segment)."""
    windows = data.get("anti_repetition", {}).get("no_repeat", {})
    segs: dict[int, dict] = {}
    for s in data["sections"]:
        if s["segment"] not in segs:
            v = dict(s.get("patterns", {}))
            v["motif"] = (s.get("motif") or {}).get("id")
            segs[s["segment"]] = v
    order = [segs[k] for k in sorted(segs)]
    out = []
    for field, w in windows.items():
        for i, cur in enumerate(order):
            for j in range(max(0, i - w), i):
                if cur.get(field) is not None and order[j].get(field) == cur.get(field):
                    out.append(f"segment {i}: {field}={cur.get(field)!r} repeats segment {j} (window {w})")
    return out
