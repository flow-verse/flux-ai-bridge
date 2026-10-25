# SPDX-License-Identifier: GPL-3.0-or-later
"""Surge XT specifics the flux-host renderer needs beyond the engine API.

- Scala tuning embedded in a patch: Surge XT applies a scale stored in a
  patch (`<patchTuning v="base64 SCL" m="base64 KBM" mname="..."/>`) when its
  user default `overrideTuningOnPatchLoad` is on. `tuned_patch` copies a
  factory `.fxp`, inserts the element and fixes the container sizes;
  `scale_slots`/`scl_text`/`kbm_text` build a 12-note-per-octave scale whose
  chromatic slots carry a maqam's cents, so plain MIDI notes sound the
  quarter tones and no pitch bend is needed. `enable_patch_tuning` writes the
  user default.
- FX slots: Surge's "FX <slot> FX Type" parameters are CLAP floats 0..1 that
  index the effect list; `fx_type_value` gives the value for an effect name
  (Surge XT 1.3.x list, 30 types).
"""
from __future__ import annotations

import base64
import os
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

# Surge XT 1.3.x fx_type order (SurgeStorage.h); CLAP exposes the type as index / (n - 1)
SURGE_FX_TYPES = [
    "Off", "Delay", "Reverb 1", "Phaser", "Rotary Speaker", "Distortion", "EQ", "Frequency Shifter",
    "Conditioner", "Chorus", "Vocoder", "Reverb 2", "Flanger", "Ring Modulator", "Airwindows", "Neuron",
    "Graphic EQ", "Resonator", "CHOW", "Exciter", "Ensemble", "Combulator", "Nimbus", "Tape", "Treemonster",
    "Waveshaper", "Mid-Side Tool", "Spring Reverb", "Bonsai", "Audio Input",
]
FX_SLOTS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "S1", "S2", "S3", "S4", "G1", "G2", "G3", "G4")

SURGE_DATA_DIRS = [
    Path.home() / "Library/Application Support/Surge XT",
    Path.home() / "Documents/Surge XT",
    Path("/Library/Application Support/Surge XT"),
    Path.home() / ".local/share/surge-xt",
    Path("/usr/share/surge-xt"),
]
USER_DEFAULTS = Path.home() / "Documents/Surge XT/SurgeXTUserDefaults.xml"
TUNING_DEFAULTS = ("overrideTuningOnPatchLoad", "overrideMappingOnPatchLoad")


def fx_type_value(name: str) -> float:
    """Normalised value of an "FX <slot> FX Type" parameter for an effect name (case-insensitive)."""
    names = [n.lower() for n in SURGE_FX_TYPES]
    key = name.strip().lower()
    if key not in names:
        raise KeyError(f"unknown Surge XT effect {name!r}; one of {', '.join(SURGE_FX_TYPES)}")
    return names.index(key) / (len(SURGE_FX_TYPES) - 1)


# ------------------------------------------------------------------ scale

def scale_slots(cents: list[float]) -> list[float]:
    """12 chromatic slots (cents above the tonic) with the scale's degrees snapped in.

    Each degree takes the slot nearest its cents (E half-flat at 150 takes the E slot);
    the other slots keep 12-TET. Two degrees wanting one slot is an error."""
    slots = [100.0 * i for i in range(12)]
    taken: dict[int, float] = {}
    for c in cents:
        within = float(c) % 1200.0
        slot = int(round(within / 100.0)) % 12
        if slot in taken and abs(taken[slot] - within) > 1e-6:
            raise ValueError(f"degrees at {taken[slot]} and {within} cents both map to chromatic slot {slot}")
        taken[slot] = within
        slots[slot] = within
    return slots


def scl_text(slots: list[float], name: str) -> str:
    lines = [f"! {name}.scl", "!", name, " 12", "!"]
    for s in slots[1:]:
        lines.append(f" {s:.5f}")
    lines.append(" 1200.00000")
    return "\n".join(lines) + "\n"


def kbm_text(tonic_midi: int, ref_hz: float) -> str:
    """Keyboard mapping: 12 keys per octave, scale degree 0 on `tonic_midi`, which sounds at `ref_hz`."""
    lines = ["! flux.kbm", "! map size", "12", "! first/last MIDI note", "0", "127",
             "! middle note (scale degree 0)", str(int(tonic_midi)), "! reference note and frequency",
             str(int(tonic_midi)), f"{ref_hz:.6f}", "! scale degree for the formal octave", "12", "! mapping"]
    lines += [str(i) for i in range(12)]
    return "\n".join(lines) + "\n"


def pitch_to_slot_note(pitch: float, tonic_midi: int, slots: list[float]) -> int:
    """MIDI note whose retuned slot sounds the (float) pitch, relative to the tonic (nearest slot)."""
    rel = (pitch - tonic_midi) * 100.0
    octave, within = divmod(rel, 1200.0)
    best = min(range(12), key=lambda i: min(abs(slots[i] - within), abs(slots[i] - within + 1200.0), abs(slots[i] - within - 1200.0)))
    # a slot just below 1200 wrapping to the next octave
    if abs(slots[best] - within + 1200.0) < abs(slots[best] - within) - 1e-9:
        octave += 1
    elif abs(slots[best] - within - 1200.0) < abs(slots[best] - within) - 1e-9:
        octave -= 1
    return int(tonic_midi + 12 * int(octave) + best)


# ------------------------------------------------------------------ patch files

def find_preset_file(preset: str) -> Path | None:
    """The .fxp behind a preset name such as "Surge XT Factory Presets/Organ 1" (case-insensitive stem)."""
    import re

    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())  # noqa: E731  "Pianobass" is the file "Piano Bass.fxp"
    stem = norm(preset.rsplit("/", 1)[-1])
    loose: Path | None = None
    for d in SURGE_DATA_DIRS:
        if not d.is_dir():
            continue
        for sub in ("patches_factory", "patches_3rdparty", "Patches", "patches_user"):
            root = d / sub
            if not root.is_dir():
                continue
            for p in root.rglob("*.fxp"):
                if p.stem.strip().lower() == preset.rsplit("/", 1)[-1].strip().lower():
                    return p
                if loose is None and norm(p.stem) == stem:
                    loose = p
    return loose


def tuned_patch(src: Path, dst: Path, scl: str, kbm: str | None = None, mapping_name: str = "flux") -> Path:
    """Copy a Surge .fxp with a patchTuning element carrying `scl` (and `kbm`) and fixed sizes."""
    raw = Path(src).read_bytes()
    if raw[:4] != b"CcnK" or raw[8:12] != b"FPCh" or raw[60:64] != b"sub3":
        raise ValueError(f"{src} is not a Surge XT .fxp patch")
    chunk_size = struct.unpack(">i", raw[56:60])[0]
    xml_size = struct.unpack("<I", raw[64:68])[0]
    head, xml, tail = raw[:92], raw[92:92 + xml_size], raw[92 + xml_size:60 + chunk_size]
    end = xml.rfind(b"</patch>")
    if end < 0:
        raise ValueError(f"{src}: no </patch> in the patch XML")
    el = ET.Element("patchTuning")
    el.set("v", base64.b64encode(scl.encode()).decode())
    if kbm:
        el.set("m", base64.b64encode(kbm.encode()).decode())
        el.set("mname", mapping_name)
    # drop an existing tuning first
    start = xml.find(b"<patchTuning")
    if start >= 0:
        close = xml.find(b"/>", start)
        xml = xml[:start] + xml[close + 2:]
        end = xml.rfind(b"</patch>")
    xml = xml[:end] + ET.tostring(el) + xml[end:]
    new_chunk = struct.pack("<I", len(xml))
    out = bytearray(head)
    out[64:68] = new_chunk
    out += xml + tail
    new_chunk_size = len(out) - 60
    out[56:60] = struct.pack(">i", new_chunk_size)
    if struct.unpack(">i", raw[4:8])[0]:
        out[4:8] = struct.pack(">i", len(out) - 8)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(out))
    return dst


def patch_xml(fxp: Path) -> bytes:
    raw = Path(fxp).read_bytes()
    xml_size = struct.unpack("<I", raw[64:68])[0]
    return raw[92:92 + xml_size]


# ------------------------------------------------------------------ user default

def patch_tuning_enabled(path: Path | None = None) -> bool:
    p = Path(path or os.environ.get("FLUX_SURGE_USER_DEFAULTS") or USER_DEFAULTS)
    if not p.is_file():
        return False
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError:
        return False
    on = {d.get("key"): d.get("value") for d in root.iter("default")}
    return all(on.get(k) == "1" for k in TUNING_DEFAULTS)


def enable_patch_tuning(path: Path | None = None) -> Path:
    """Set Surge XT's overrideTuningOnPatchLoad / overrideMappingOnPatchLoad to 1 (keeps other defaults)."""
    p = Path(path or os.environ.get("FLUX_SURGE_USER_DEFAULTS") or USER_DEFAULTS)
    entries: list[tuple[str, dict]] = []
    if p.is_file():
        root = ET.parse(p).getroot()
        for d in root.iter("default"):
            if d.get("key") not in TUNING_DEFAULTS:
                entries.append(("default", dict(d.attrib)))
    for k in TUNING_DEFAULTS:
        entries.append(("default", {"key": k, "value": "1", "type": "2"}))
    lines = ['<?xml version = "1.0" encoding = "UTF-8" ?>', '<defaults version="1">']
    for _, attrs in entries:
        lines.append("  <default " + " ".join(f'{a}="{v}"' for a, v in attrs.items()) + "/>")
    lines.append("</defaults>")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--enable" in sys.argv:
        print(enable_patch_tuning())
    else:
        print("patch tuning enabled:", patch_tuning_enabled(), "(", USER_DEFAULTS, ")")
