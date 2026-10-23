# SPDX-License-Identifier: GPL-3.0-or-later
"""Engine supervision with a fake engine: crash, timeout, stray output, restart; and the stdio entry point."""
import os
import stat
import sys
from pathlib import Path

import pytest

from flux_mcp.engine import EngineClient, EngineCrashed, EngineError, EngineNotFound, find_engine

FAKE = Path(__file__).with_name("fake_engine.py")


@pytest.fixture
def fake_engine(tmp_path) -> str:
    exe = tmp_path / "flux-engine"
    exe.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{FAKE}' \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def test_find_engine(fake_engine, monkeypatch):
    assert find_engine(fake_engine) == fake_engine
    monkeypatch.setenv("FLUX_ENGINE", fake_engine)
    assert find_engine() == fake_engine
    with pytest.raises(EngineNotFound):
        find_engine("/nonexistent/flux-engine")


def test_calls_errors_and_noise(fake_engine):
    e = EngineClient(fake_engine)
    try:
        assert e.call("ping") == {"pong": True}
        assert e.call("noise") == {"pong": True}  # a non-JSON line on stdout is skipped
        with pytest.raises(EngineError) as ei:
            e.call("frobnicate")
        assert ei.value.code == -32601
        assert e.sample_rate == 48000
    finally:
        e.close()


def test_crash_then_restart(fake_engine):
    e = EngineClient(fake_engine)
    try:
        e.call("ping")
        with pytest.raises(EngineCrashed) as ei:
            e.call("crash")
        assert "restarted on the next call" in str(ei.value)
        assert "fake engine: crashing" in str(ei.value) or "exited" in str(ei.value)
        assert e.call("ping") == {"pong": True}
        assert e.restarts == 1 and "restarted" in (e.notice or "")
    finally:
        e.close()


def test_timeout_kills_and_restarts(fake_engine):
    e = EngineClient(fake_engine)
    try:
        with pytest.raises(EngineCrashed) as ei:
            e.call("hang", timeout=0.5)
        assert "did not answer 'hang' within" in str(ei.value)
        assert e.call("ping") == {"pong": True}
    finally:
        e.close()


@pytest.mark.anyio
async def test_stdio_server_speaks_only_mcp(fake_engine, tmp_path):
    """Run the real `flux-mcp` entry point over stdio; any stray stdout would break the client."""
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "flux_mcp", "--engine", fake_engine],
                                   env={"PYTHONPATH": os.pathsep.join(sys.path)}, cwd=str(tmp_path))
    async with Client(params) as c:
        tools = (await c.list_tools()).tools
        assert len(tools) >= 15
        r = await c.call_tool("get_project", {})
        # the fake engine knows no describe_project: the error must come back as a tool error, not a crash
        assert r.is_error and "unknown method" in r.content[0].text
