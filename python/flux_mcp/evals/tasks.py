# SPDX-License-Identifier: GPL-3.0-or-later
"""The eval tasks: prompt, setup, reference tool calls and checker."""
from __future__ import annotations

import json
import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

TEST_SYNTH = "Test Synth"


class ToolCallError(Exception):
    pass


@dataclass
class Ctx:
    """What setup, reference and checker functions get."""
    client: Any                      # mcp.Client connected to the flux-mcp server
    session: Any                     # flux_mcp.server.Session (checkers read the engine directly)
    dir: Path                        # scratch directory for this task
    facts: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    async def tool(self, tool_name: str, /, **args) -> dict:
        r = await self.client.call_tool(tool_name, args)
        self.calls.append(tool_name)
        text = r.content[0].text if r.content else ""
        if r.is_error:
            raise ToolCallError(f"{tool_name}: {text}")
        return r.structured_content or json.loads(text or "{}")

    # -- checker helpers (read the engine directly, never through the agent's tools) --
    def project(self, notes: bool = True) -> dict:
        p = self.session.describe(include_notes=notes)
        if p is None:
            raise AssertionError("no project is open")
        return p


# ------------------------------------------------------------------- helpers
def tracks(p: dict, name_part: str | None = None, kind: str | None = None) -> list[dict]:
    out = []
    for t in p.get("tracks", []):
        if t["type"] in ("chord", "modulator", "marker", "master"):
            continue
        if name_part and name_part.lower() not in t["name"].lower():
            continue
        if kind and t["type"] != kind:
            continue
        out.append(t)
    return out


def bpm_at(p: dict, bar: float) -> float:
    bpm = p["tempo"]["bpm"]
    for ch in sorted(p["tempo"].get("changes", []), key=lambda c: c["bar"]):
        if ch["bar"] <= bar + 1e-9:
            bpm = ch["bpm"]
    return bpm


def beats_per_bar(p: dict) -> int:
    return p["time_signature"]["numerator"]


def abs_notes(p: dict, t: dict) -> list[dict]:
    """Every note of a track with an absolute beat position (0 = start of bar 1)."""
    bpb = beats_per_bar(p)
    out = []
    for c in t.get("clips", []):
        if c.get("kind") != "midi":
            continue
        for n in c.get("notes", []):
            out.append({**n, "abs": (c["start_bar"] - 1) * bpb + n["start_beat"], "clip": c["id"]})
    return sorted(out, key=lambda n: (n["abs"], n["pitch"]))


def clip_signature(c: dict, transpose: int = 0) -> list[tuple]:
    return sorted((round(n["start_beat"], 4), round(n["length_beats"], 4), n["pitch"] + transpose) for n in c.get("notes", []))


def flac_duration(path: Path) -> float:
    with open(path, "rb") as f:
        if f.read(4) != b"fLaC":
            raise AssertionError(f"{path} is not a FLAC file")
        f.read(4)  # metadata block header (STREAMINFO first)
        si = f.read(34)
    sr = (si[10] << 12) | (si[11] << 4) | (si[12] >> 4)
    total = ((si[13] & 0x0F) << 32) | int.from_bytes(si[14:18], "big")
    return total / sr


def wav_duration(path: Path) -> float:
    data = path.read_bytes()
    if data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise AssertionError(f"{path} is not a WAV file")
    pos, sr, block, size = 12, None, None, None
    while pos + 8 <= len(data):
        cid, n = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, block, _ = struct.unpack("<HHIIHH", data[pos + 8:pos + 24])
        elif cid == b"data":
            size = n
            break
        pos += 8 + n + (n & 1)
    if not (sr and block and size is not None):
        raise AssertionError(f"{path}: no fmt/data chunk")
    return size / block / sr


def write_test_midi(path: Path, pitches: list[int], ppq: int = 480) -> None:
    """A minimal type-0 SMF: one quarter note per pitch at 120 bpm."""
    ev = bytearray()
    ev += b"\x00\xff\x51\x03" + (500000).to_bytes(3, "big")
    for p in pitches:
        ev += b"\x00" + bytes([0x90, p, 100])
        ev += _vlq(ppq) + bytes([0x80, p, 0])
    ev += b"\x00\xff\x2f\x00"
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq) + b"MTrk" + struct.pack(">I", len(ev)) + bytes(ev))


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_test_wav(path: Path, seconds: float = 1.0, sr: int = 48000) -> None:
    frames = bytearray()
    for i in range(int(seconds * sr)):
        v = int(0.3 * 32767 * math.sin(2 * math.pi * 220 * i / sr))
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def note(pitch, start, length=1.0, vel=100) -> dict:
    return {"pitch": pitch, "start_beat": start, "length_beats": length, "velocity": vel}


Check = Callable[[Ctx], None]   # raises AssertionError with a reason on failure


@dataclass
class Task:
    id: str
    prompt: str
    reference: Callable[[Ctx], Awaitable[None]]
    check: Check
    setup: Callable[[Ctx], Awaitable[None]] | None = None
    bpm: float = 120.0
    needs_ai: bool = False      # needs the flux-ai worker (sine-test backend)

    def render_prompt(self, ctx: Ctx) -> str:
        return self.prompt.format(dir=ctx.dir, **ctx.facts)


async def _track(ctx: Ctx, name: str, kind: str = "instrument", plugin: str | None = TEST_SYNTH) -> str:
    args: dict = {"kind": kind, "name": name}
    if plugin and kind == "instrument":
        args["plugin"] = plugin
    return (await ctx.tool("add_track", **args))["track"]


# =================================================================== tasks
TASKS: list[Task] = []


def task(**kw):
    def deco(ref):
        TASKS.append(Task(reference=ref, **kw))
        return ref
    return deco


# 1 -------------------------------------------------------------------------
def _check_groove(ctx: Ctx) -> None:
    p = ctx.project()
    assert abs(bpm_at(p, 1) - 124) < 0.01, f"tempo is {bpm_at(p, 1)}, expected 124"
    drums, bass = tracks(p, "drum"), tracks(p, "bass")
    assert drums and bass, "need a Drums track and a Bass track"
    dn = [n for t in drums for n in abs_notes(p, t)]
    bn = [n for t in bass for n in abs_notes(p, t)]
    assert len(dn) >= 8, f"drums have {len(dn)} notes, expected at least 8"
    assert len(bn) >= 4, f"bass has {len(bn)} notes, expected at least 4"
    for n in dn + bn:
        assert n["abs"] + n["length_beats"] <= 16 + 1e-6, "notes must stay within bars 1-4"
    assert all(t["plugins"] for t in drums + bass), "tracks need an instrument plugin"


@task(id="drum_bass_groove_124", check=_check_groove,
      prompt="Create a 4-bar drum and bass groove at 124 BPM starting at bar 1: a Drums track and a Bass track, "
             "both using the Test Synth instrument.")
async def _ref_groove(ctx: Ctx) -> None:
    await ctx.tool("set_tempo", bpm=124)
    d = await _track(ctx, "Drums")
    kick = [note(36, b, 0.5, 110) for b in range(16)]
    snare = [note(38, b, 0.5, 100) for b in range(1, 16, 2)]
    hats = [note(42, b + 0.5, 0.25, 70) for b in range(16)]
    await ctx.tool("write_midi_clip", track_id=d, start_bar=1, length_bars=4, notes=kick + snare + hats, name="Drums")
    b = await _track(ctx, "Bass")
    line = [note("C2", 0, 0.75), note("C2", 1.5, 0.5), note("Eb2", 3, 1), note("G1", 4, 1.5), note("Bb1", 6, 1),
            note("C2", 8, 0.75), note("C2", 9.5, 0.5), note("F2", 11, 1), note("G2", 12, 2), note("Bb1", 14.5, 1.5)]
    await ctx.tool("write_midi_clip", track_id=b, start_bar=1, length_bars=4, notes=line, name="Bass")


# 2 -------------------------------------------------------------------------
C_MINOR = {0, 2, 3, 5, 7, 8, 10, 11}


def _check_pad(ctx: Ctx) -> None:
    p = ctx.project()
    pads = tracks(p, "pad")
    assert pads, "no Pad track"
    ns = [n for t in pads for n in abs_notes(p, t)]
    onsets: dict[float, set[int]] = {}
    for n in ns:
        onsets.setdefault(round(n["abs"], 3), set()).add(n["pitch"])
    chords = {k: v for k, v in onsets.items() if len(v) >= 3}
    assert len(chords) >= 4, f"found {len(chords)} chords of 3+ notes, expected 4"
    first = chords[min(chords)]
    assert {0, 3, 7} <= {x % 12 for x in first}, "the first chord should be C minor (C, Eb, G)"
    stray = {n["pitch"] % 12 for n in ns} - C_MINOR
    assert not stray, f"pitch classes outside C minor: {sorted(stray)}"
    assert max(n["abs"] + n["length_beats"] for n in ns) <= 16 + 1e-6, "progression should fit in 4 bars"


@task(id="pad_c_minor_progression", check=_check_pad,
      prompt="Add a Pad track using the Test Synth with a 4-bar chord progression in C minor starting at bar 1: one "
             "chord per bar, each chord with at least three notes, starting on C minor.")
async def _ref_pad(ctx: Ctx) -> None:
    t = await _track(ctx, "Pad")
    chords = [["C3", "Eb3", "G3"], ["Ab2", "C3", "Eb3"], ["Eb3", "G3", "Bb3"], ["Bb2", "D3", "F3"]]
    notes = [note(pn, i * 4, 4, 80) for i, ch in enumerate(chords) for pn in ch]
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=4, notes=notes, name="Pad")


# 3 -------------------------------------------------------------------------
BASSLINE = [note(36, 0, 1), note(36, 1.5, 0.5), note(39, 2, 1), note(43, 3, 1), note(41, 4, 2), note(39, 6, 2)]


async def _setup_bassline(ctx: Ctx) -> None:
    t = await _track(ctx, "Bass")
    r = await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=2, notes=BASSLINE, name="Bassline")
    ctx.facts.update(track=t, clip=r["clip"])


def _check_transpose(ctx: Ctx) -> None:
    p = ctx.project()
    want = clip_signature({"notes": BASSLINE}, 12)
    for t in tracks(p):
        for c in t.get("clips", []):
            if c["id"] != ctx.facts["clip"] and c.get("kind") == "midi" and clip_signature(c) == want:
                return
    raise AssertionError("no clip with the bassline transposed up 12 semitones")


@task(id="transpose_bass_octave", setup=_setup_bassline, check=_check_transpose,
      prompt="Transpose the bassline on the Bass track up an octave. Clips cannot be edited in place yet, so write "
             "the transposed version as a new clip right after the original.")
async def _ref_transpose(ctx: Ctx) -> None:
    await ctx.tool("copy_clip", clip_id=ctx.facts["clip"], to_bar=3, transpose_semitones=12)


# 4 -------------------------------------------------------------------------
async def _setup_8_bars(ctx: Ctx) -> None:
    t = await _track(ctx, "Keys")
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=8,
                   notes=[note(60 + (b % 5), b, 1) for b in range(32)])
    ctx.facts["out"] = str(ctx.dir / "bars1-8.flac")


def _check_flac(ctx: Ctx) -> None:
    out = Path(ctx.facts["out"])
    assert out.is_file(), f"{out} was not written"
    dur = flac_duration(out)
    assert abs(dur - 16.0) < 0.05, f"{out.name} is {dur:.3f} s; bars 1-8 at 120 bpm are 16.0 s"


@task(id="render_bars_1_8_flac", setup=_setup_8_bars, check=_check_flac,
      prompt="Render bars 1-8 of the song to FLAC at {out}.")
async def _ref_flac(ctx: Ctx) -> None:
    await ctx.tool("render_mix", out=ctx.facts["out"], start_bar=1, end_bar=9, format="flac")


# 5 -------------------------------------------------------------------------
def _check_tempo_90(ctx: Ctx) -> None:
    p = ctx.project(False)
    assert abs(bpm_at(p, 1) - 90) < 0.01 and abs(bpm_at(p, 100) - 90) < 0.01, f"tempo at bar 1 is {bpm_at(p, 1)}"


@task(id="set_tempo_90", check=_check_tempo_90, prompt="Set the tempo of the whole song to 90 BPM.")
async def _ref_tempo(ctx: Ctx) -> None:
    await ctx.tool("set_tempo", bpm=90)


# 6 -------------------------------------------------------------------------
def _check_tempo_change(ctx: Ctx) -> None:
    p = ctx.project(False)
    assert abs(bpm_at(p, 1) - 120) < 0.01, f"the start should stay at 120 bpm, is {bpm_at(p, 1)}"
    assert abs(bpm_at(p, 16.9) - 120) < 0.01, "bars before 17 should stay at 120 bpm"
    assert abs(bpm_at(p, 17) - 128) < 0.01, f"tempo from bar 17 is {bpm_at(p, 17)}, expected 128"


@task(id="tempo_change_bar_17", check=_check_tempo_change,
      prompt="From bar 17 onwards the tempo should be 128 BPM; keep everything before bar 17 at the current tempo.")
async def _ref_tempo_change(ctx: Ctx) -> None:
    await ctx.tool("set_tempo", bpm=128, at_bar=17)


# 7 -------------------------------------------------------------------------
def _check_markers(ctx: Ctx) -> None:
    got = {(m["name"].strip().lower(), round(m["bar"], 3)) for m in ctx.project(False).get("markers", [])}
    want = {("intro", 1.0), ("drop", 17.0), ("outro", 33.0)}
    assert want <= got, f"missing markers: {sorted(want - got)}"


@task(id="markers_song_form", check=_check_markers,
      prompt="Add arrangement markers: Intro at bar 1, Drop at bar 17 and Outro at bar 33.")
async def _ref_markers(ctx: Ctx) -> None:
    for name, bar in (("Intro", 1), ("Drop", 17), ("Outro", 33)):
        await ctx.tool("add_marker", name=name, bar=bar)


# 8 -------------------------------------------------------------------------
def _check_range(ctx: Ctx) -> None:
    r = ctx.project(False)["range"]
    assert abs(r["start_bar"] - 1) < 1e-6 and abs(r["end_bar"] - 33) < 1e-6, f"range is {r['start_bar']}-{r['end_bar']}, expected 1-33"


@task(id="song_range_32_bars", check=_check_range,
      prompt="Make the song exactly 32 bars long starting at bar 1, so a default render covers those 32 bars.")
async def _ref_range(ctx: Ctx) -> None:
    await ctx.tool("set_render_range", start_bar=1, end_bar=33)


# 9 -------------------------------------------------------------------------
A_MINOR = {9, 11, 0, 2, 4, 5, 7}


def _check_lead(ctx: Ctx) -> None:
    p = ctx.project()
    leads = tracks(p, "lead")
    assert leads and leads[0]["plugins"], "need a Lead track with an instrument"
    ns = [n for t in leads for n in abs_notes(p, t)]
    assert len(ns) == 8, f"melody has {len(ns)} notes, expected 8"
    assert all(16 - 1e-6 <= n["abs"] and n["abs"] + n["length_beats"] <= 24 + 1e-6 for n in ns), "notes must lie in bars 5-6"
    stray = {n["pitch"] % 12 for n in ns} - A_MINOR
    assert not stray, f"notes outside A minor: {sorted(stray)}"


@task(id="lead_melody_a_minor", check=_check_lead,
      prompt="Add a Lead track using the Test Synth with an 8-note melody in A minor lasting 2 bars, starting at bar 5.")
async def _ref_lead(ctx: Ctx) -> None:
    t = await _track(ctx, "Lead")
    mel = ["A4", "C5", "E5", "D5", "C5", "B4", "G4", "A4"]
    await ctx.tool("write_midi_clip", track_id=t, start_bar=5, length_bars=2,
                   notes=[note(pn, i, 1, 96) for i, pn in enumerate(mel)], name="Lead")


# 10/11 ---------------------------------------------------------------------
async def _setup_bass_synth(ctx: Ctx) -> None:
    r = await ctx.tool("add_track", kind="instrument", name="Bass", plugin=TEST_SYNTH)
    plug = r["results"][0]["plugin"]["id"]
    ctx.facts.update(track=r["track"], plugin=plug)


def _param(ctx: Ctx, name: str) -> dict:
    ps = ctx.session.call("list_params", {"plugin": ctx.facts["plugin"], "include_hidden": True})["params"]
    return next(p for p in ps if p["name"] == name)


def _check_level(ctx: Ctx) -> None:
    v = _param(ctx, "Level")["plain_value"]
    assert abs(v - 0.5) < 0.01, f"Level is {v}, expected 0.5"


@task(id="bass_level_half", setup=_setup_bass_synth, check=_check_level,
      prompt="Set the Level parameter of the Bass track's synth to 0.5.")
async def _ref_level(ctx: Ctx) -> None:
    await ctx.tool("set_plugin_params", plugin_id=ctx.facts["plugin"], values=[{"name": "Level", "value": 0.5, "unit": "plain"}])


def _check_gain(ctx: Ctx) -> None:
    v = _param(ctx, "Gain")["plain_value"]
    db = 20 * math.log10(v) if v > 0 else -999
    assert abs(db + 6) < 0.3, f"Gain is {v:.4f} ({db:.2f} dB), expected -6 dB"


@task(id="bass_gain_minus_6db", setup=_setup_bass_synth, check=_check_gain,
      prompt="Turn the Bass track's synth down by 6 dB using the plugin's Gain parameter (it is at 0 dB now).")
async def _ref_gain(ctx: Ctx) -> None:
    await ctx.tool("set_plugin_params", plugin_id=ctx.facts["plugin"], values=[{"name": "Gain", "value": -6, "unit": "db"}])


# 12 ------------------------------------------------------------------------
async def _setup_two_clips(ctx: Ctx) -> None:
    t = await _track(ctx, "Keys")
    a = await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=1, notes=[note(60, 0, 4)])
    b = await ctx.tool("write_midi_clip", track_id=t, start_bar=9, length_bars=1, notes=[note(72, 0, 4)])
    ctx.facts.update(keep=a["clip"], gone=b["clip"])


def _check_undo(ctx: Ctx) -> None:
    ids = {c["id"] for t in tracks(ctx.project(False)) for c in t.get("clips", [])}
    assert ctx.facts["gone"] not in ids, "the clip at bar 9 is still there"
    assert ctx.facts["keep"] in ids, "the clip at bar 1 was removed too"


@task(id="undo_last_change", setup=_setup_two_clips, check=_check_undo,
      prompt="The clip just added at bar 9 was a mistake. Undo that last change, and nothing else.")
async def _ref_undo(ctx: Ctx) -> None:
    await ctx.tool("undo_last_turn")


# 13 ------------------------------------------------------------------------
async def _setup_two_tracks(ctx: Ctx) -> None:
    for name, pitch in (("Bass", 36), ("Keys", 60)):
        t = await _track(ctx, name)
        await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=4, notes=[note(pitch, b, 1) for b in range(16)])
    ctx.facts["out"] = str(ctx.dir / "mix.wav")


def _check_stems(ctx: Ctx) -> None:
    out = Path(ctx.facts["out"])
    assert out.is_file(), f"{out} was not written"
    assert abs(wav_duration(out) - 8.0) < 0.05, f"mix is {wav_duration(out):.3f} s, expected 8.0 (bars 1-4 at 120 bpm)"
    stems = sorted(out.parent.glob(f"{out.stem} - *.wav"))
    assert len(stems) >= 2, f"found {len(stems)} stem files, expected one per track (2)"


@task(id="render_stems", setup=_setup_two_tracks, check=_check_stems,
      prompt="Render bars 1-4 to the WAV file {out}, and also write one stem file per track.")
async def _ref_stems(ctx: Ctx) -> None:
    await ctx.tool("render_mix", out=ctx.facts["out"], start_bar=1, end_bar=5, stems=True)


# 14 ------------------------------------------------------------------------
def _check_four_floor(ctx: Ctx) -> None:
    p = ctx.project()
    ds = tracks(p, "drum")
    assert ds, "no Drums track"
    kicks = [n for t in ds for n in abs_notes(p, t) if n["pitch"] == 36]
    starts = sorted(round(n["abs"], 3) for n in kicks)
    assert starts == [float(b) for b in range(32)], f"expected 32 kicks on beats 0-31, got {len(starts)}"


@task(id="four_on_the_floor", check=_check_four_floor,
      prompt="Create a Drums track using the Test Synth and write a four-on-the-floor kick (MIDI note 36, one hit on "
             "every beat) for 8 bars starting at bar 1.")
async def _ref_four_floor(ctx: Ctx) -> None:
    t = await _track(ctx, "Drums")
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=8, notes=[note(36, b, 0.5, 115) for b in range(32)])


# 15 ------------------------------------------------------------------------
def _check_arp(ctx: Ctx) -> None:
    p = ctx.project()
    arps = tracks(p, "arp")
    assert arps, "no Arp track"
    ns = [n for t in arps for n in abs_notes(p, t)]
    assert sorted(round(n["abs"], 3) for n in ns) == [i * 0.25 for i in range(16)], "expected 16 sixteenth notes in bar 1"
    pcs = {n["pitch"] % 12 for n in ns}
    assert pcs == {9, 0, 4}, f"pitch classes {sorted(pcs)}, expected A, C, E"


@task(id="arpeggio_16ths", check=_check_arp,
      prompt="On a new Arp track using the Test Synth, write a one-bar 16th-note arpeggio of an A minor triad "
             "starting at bar 1.")
async def _ref_arp(ctx: Ctx) -> None:
    t = await _track(ctx, "Arp")
    seq = ["A3", "C4", "E4", "A4"]
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=1,
                   notes=[note(seq[i % 4], i * 0.25, 0.25, 90) for i in range(16)])


# 16 ------------------------------------------------------------------------
def _check_sections(ctx: Ctx) -> None:
    p = ctx.project()
    ks = tracks(p, "keys")
    assert ks, "no Keys track"
    clips = sorted((c for t in ks for c in t.get("clips", [])), key=lambda c: c["start_bar"])
    starts = {(round(c["start_bar"], 3), round(c["end_bar"], 3)) for c in clips}
    assert {(1.0, 9.0), (9.0, 17.0)} <= starts, f"clips span {sorted(starts)}, expected bars 1-9 and 9-17"
    assert all(c.get("note_count", len(c.get("notes", []))) >= 4 for c in clips), "each clip needs at least 4 notes"


@task(id="two_sections_keys", check=_check_sections,
      prompt="Create a Keys track using the Test Synth with two 8-bar clips: an intro clip starting at bar 1 and a "
             "chorus clip starting at bar 9. Each clip needs at least 4 notes.")
async def _ref_sections(ctx: Ctx) -> None:
    t = await _track(ctx, "Keys")
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=8, name="Intro",
                   notes=[note(p, i * 8, 8, 70) for i, p in enumerate(["A3", "F3", "C4", "G3"])])
    await ctx.tool("write_midi_clip", track_id=t, start_bar=9, length_bars=8, name="Chorus",
                   notes=[note(p, i * 4, 4, 90) for i, p in enumerate(["A4", "F4", "C5", "G4"] * 2)])


# 17 ------------------------------------------------------------------------
async def _setup_saveas(ctx: Ctx) -> None:
    t = await _track(ctx, "Keys")
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=1, notes=[note(60, 0, 4)])
    ctx.facts["target"] = str(ctx.dir / "saved-copy")


def _check_saveas(ctx: Ctx) -> None:
    target = Path(ctx.facts["target"])
    assert target.is_dir() and any(target.iterdir()), f"{target} was not written"
    p = ctx.project(False)
    assert Path(p["dir"]) == target, f"the open project is {p['dir']}, expected {target}"
    assert not ctx.session.dirty(), "there are still unsaved changes"


@task(id="save_as", setup=_setup_saveas, check=_check_saveas, prompt="Save the project as {target}.")
async def _ref_saveas(ctx: Ctx) -> None:
    await ctx.tool("save_project", dir=ctx.facts["target"])


# 18 ------------------------------------------------------------------------
async def _setup_newdir(ctx: Ctx) -> None:
    ctx.facts["target"] = str(ctx.dir / "waltz")


def _check_new_project(ctx: Ctx) -> None:
    p = ctx.project(False)
    assert Path(p["dir"]) == Path(ctx.facts["target"]), f"open project is {p['dir']}"
    ts = p["time_signature"]
    assert (ts["numerator"], ts["denominator"]) == (6, 8), f"time signature is {ts['numerator']}/{ts['denominator']}"
    assert abs(bpm_at(p, 1) - 100) < 0.01, f"tempo is {bpm_at(p, 1)}"


@task(id="new_project_6_8", setup=_setup_newdir, check=_check_new_project,
      prompt="Start a new project in {target} at 100 BPM in 6/8 time.")
async def _ref_new_project(ctx: Ctx) -> None:
    await ctx.tool("new_project", dir=ctx.facts["target"], bpm=100, time_signature="6/8")


# 19 ------------------------------------------------------------------------
async def _setup_midi_file(ctx: Ctx) -> None:
    path = ctx.dir / "riff.mid"
    write_test_midi(path, [57, 60, 64, 67])
    ctx.facts["path"] = str(path)


def _check_import_midi(ctx: Ctx) -> None:
    p = ctx.project()
    clips = [c for t in tracks(p) for c in t.get("clips", []) if c.get("kind") == "midi"]
    assert any(abs(c["start_bar"] - 5) < 1e-6 and c.get("note_count", 0) >= 4 for c in clips), \
        f"no MIDI clip with the riff at bar 5 (clips at {[c['start_bar'] for c in clips]})"


@task(id="import_midi_bar_5", setup=_setup_midi_file, check=_check_import_midi,
      prompt="Import the MIDI file {path} into the song at bar 5.")
async def _ref_import_midi(ctx: Ctx) -> None:
    await ctx.tool("import_file", path=ctx.facts["path"], start_bar=5)


# 20 ------------------------------------------------------------------------
async def _setup_wav_file(ctx: Ctx) -> None:
    path = ctx.dir / "vocal-chop.wav"
    write_test_wav(path)
    ctx.facts["path"] = str(path)


def _check_import_audio(ctx: Ctx) -> None:
    p = ctx.project(False)
    clips = [c for t in tracks(p, kind="audio") for c in t.get("clips", [])]
    assert any(abs(c["start_bar"] - 3) < 1e-6 for c in clips), f"no audio clip at bar 3 (clips at {[c['start_bar'] for c in clips]})"


@task(id="import_audio_bar_3", setup=_setup_wav_file, check=_check_import_audio,
      prompt="Import the audio file {path} into the song at bar 3.")
async def _ref_import_audio(ctx: Ctx) -> None:
    await ctx.tool("import_file", path=ctx.facts["path"], start_bar=3)


# 21 ------------------------------------------------------------------------
def _check_generate(ctx: Ctx) -> None:
    p = ctx.project(False)
    clips = [c for t in tracks(p, kind="audio") for c in t.get("clips", [])]
    assert any(abs(c["start_bar"] - 1) < 1e-6 for c in clips), "no generated audio clip at bar 1"
    side = list(Path(p["dir"]).rglob("*.provenance.json")) + list(ctx.dir.rglob("*.provenance.json"))
    assert side, "no provenance sidecar was written"
    rec = json.loads(side[0].read_text())
    assert rec.get("ai_generated") is True and rec.get("backend") == "sine-test", "provenance does not mark sine-test output"


@task(id="generate_sine_and_import", check=_check_generate, needs_ai=True,
      prompt="Using the 'sine-test' generation backend, generate 4 seconds of test audio and put it in the song at bar 1.")
async def _ref_generate(ctx: Ctx) -> None:
    await ctx.tool("generate_audio", prompt="test tone", seconds=4, backend="sine-test", start_bar=1)


# 22 ------------------------------------------------------------------------
def _check_hats(ctx: Ctx) -> None:
    p = ctx.project()
    hs = tracks(p, "hat")
    assert hs, "no Hats track"
    ns = [n for t in hs for n in abs_notes(p, t) if n["pitch"] == 42]
    assert sorted(round(n["abs"], 3) for n in ns) == [i * 0.5 for i in range(8)], "expected 8 eighth-note hats in bar 1"
    on = [n["velocity"] for n in ns if abs(n["abs"] - round(n["abs"])) < 1e-6]
    off = [n["velocity"] for n in ns if abs(n["abs"] - round(n["abs"])) >= 1e-6]
    assert min(on) > max(off), f"on-beats {on} should be louder than off-beats {off}"


@task(id="hihat_velocity_accents", check=_check_hats,
      prompt="Create a Hats track using the Test Synth with one bar of eighth-note hi-hats (MIDI note 42) at bar 1. "
             "Accent the on-beats (velocity 110) and play the off-beats softer (velocity 70).")
async def _ref_hats(ctx: Ctx) -> None:
    t = await _track(ctx, "Hats")
    await ctx.tool("write_midi_clip", track_id=t, start_bar=1, length_bars=1,
                   notes=[note(42, i * 0.5, 0.25, 110 if i % 2 == 0 else 70) for i in range(8)])


TASKS_BY_ID = {t.id: t for t in TASKS}
