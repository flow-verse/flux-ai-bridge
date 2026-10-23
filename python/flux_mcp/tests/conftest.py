# SPDX-License-Identifier: GPL-3.0-or-later
import os
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()


def _find_up(rel: str) -> Path | None:
    """Look for <ancestor>/<rel> (e.g. a sibling flux-host build next to this repo)."""
    for parent in HERE.parents:
        cand = parent / rel
        if cand.exists():
            return cand
    return None


def engine_path() -> str | None:
    if os.environ.get("FLUX_ENGINE"):
        return os.environ["FLUX_ENGINE"]
    found = shutil.which("flux-engine")
    if found:
        return found
    p = _find_up("flux-host/conanbuild/Release/products/bin/flux-engine")
    return str(p) if p else None


def clap_path() -> str | None:
    if os.environ.get("FLUX_CLAP_PATH"):
        return os.environ["FLUX_CLAP_PATH"]
    p = _find_up("flux-host/conanbuild/Release/tests/test_plugins/CLAP/Release")
    return str(p) if p else None


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def real_engine_env():
    eng = engine_path()
    if not eng or not os.access(eng, os.X_OK):
        pytest.skip("flux-engine not found: set FLUX_ENGINE (build flux-host, target flux-engine)")
    clap = clap_path()
    if not clap:
        pytest.skip("CLAP 'Test Synth' not found: set FLUX_CLAP_PATH to flux-host's tests/test_plugins/CLAP/Release")
    return {"FLUX_ENGINE": eng, "FLUX_CLAP_PATH": clap}


@pytest.fixture
def engine_env(real_engine_env, monkeypatch, tmp_path):
    for k, v in real_engine_env.items():
        monkeypatch.setenv(k, v)
    # keep licence acceptances out of the user's real cache
    monkeypatch.setenv("FLUX_AI_CACHE", str(tmp_path / "flux-ai-cache"))
    return real_engine_env
