# SPDX-License-Identifier: GPL-3.0-or-later
"""Load the editable genre rule tables (genres/*.yaml)."""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

GENRE_DIR = Path(__file__).with_name("genres")


def _dirs() -> list[Path]:
    """Built-in tables plus any directory in $FLUX_LONGFORM_GENRES (owner overrides first)."""
    extra = [Path(p) for p in os.environ.get("FLUX_LONGFORM_GENRES", "").split(os.pathsep) if p]
    return extra + [GENRE_DIR]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


@lru_cache(maxsize=None)
def _load_file(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_file"] = Path(path).name
    data["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return data


def available() -> list[str]:
    names = []
    for d in _dirs():
        for p in sorted(d.glob("*.yaml")):
            names.append(_load_file(str(p))["name"])
    return sorted(set(names))


def load(name: str) -> dict:
    """Find a genre table by name, alias or file stem (case/punctuation-insensitive)."""
    want = _norm(name)
    for d in _dirs():
        for p in sorted(d.glob("*.yaml")):
            g = _load_file(str(p))
            keys = {_norm(g["name"]), _norm(p.stem)} | {_norm(a) for a in g.get("aliases", [])}
            if want in keys:
                return g
    raise KeyError(f"unknown genre {name!r}; available: {', '.join(available())}")


def digest(genre: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in genre.items() if not k.startswith("_")},
                                     sort_keys=True).encode()).hexdigest()
