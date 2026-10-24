# SPDX-License-Identifier: GPL-3.0-or-later
"""The eval suite: scripted runner (must pass 100%), checkers reject untouched projects, LLM runner plumbing."""
import os
from types import SimpleNamespace

import pytest

from flux_mcp.evals.runner import _check, _prepare, run_suite, summary
from flux_mcp.evals.tasks import TASKS
from flux_mcp.server import Config, Session, build_server

pytestmark = pytest.mark.anyio


def test_suite_size_and_ids():
    assert 18 <= len(TASKS) <= 25
    assert len({t.id for t in TASKS}) == len(TASKS)


async def test_scripted_suite_passes(engine_env, tmp_path):
    results = await run_suite("scripted", base=tmp_path, config=Config())
    rep = summary(results)
    failed = [(r.id, r.reason) for r in results if not r.passed]
    assert rep["pass_rate"] == 1.0, failed


async def test_checkers_reject_untouched_projects(engine_env, tmp_path):
    """Negative control: with only the setup applied, every checker must fail."""
    from mcp import Client

    session = Session(Config(workdir=tmp_path))
    async with Client(build_server(session)) as client:
        for task in TASKS:
            ctx = await _prepare(task, client, session, tmp_path)
            ok, _ = _check(task, ctx)
            assert not ok, f"{task.id}: checker passes without any work"


class _FakeMessages:
    """Scripted stand-in for the Messages API: one tool call, then end_turn."""

    def __init__(self):
        self.calls = 0

    async def create(self, **kw):
        self.calls += 1
        assert kw["model"] == "test-model" and kw["tools"] and kw["tools"][0]["input_schema"]["type"] == "object"
        if self.calls == 1:
            block = SimpleNamespace(type="tool_use", id="tu_1", name="set_tempo", input={"bpm": 90})
            return SimpleNamespace(model="test-model", stop_reason="tool_use", content=[block])
        last = kw["messages"][-1]["content"][0]
        assert last["type"] == "tool_result" and not last["is_error"]
        return SimpleNamespace(model="test-model", stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")])


async def test_llm_runner_plumbing(engine_env, tmp_path):
    fake = SimpleNamespace(messages=_FakeMessages())
    results = await run_suite("llm", ["set_tempo_90"], base=tmp_path, config=Config(), model="test-model",
                              llm_client=fake)
    assert results[0].passed, results[0].reason
    assert results[0].tool_calls == ["set_tempo"] and results[0].turns == 2


@pytest.mark.skipif(not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
                    reason="LLM eval needs ANTHROPIC_API_KEY; run `python -m flux_mcp.evals --llm` to measure")
async def test_llm_suite_smoke(engine_env, tmp_path):  # pragma: no cover - needs network and credentials
    results = await run_suite("llm", ["set_tempo_90", "markers_song_form"], base=tmp_path, config=Config(),
                              model=os.environ.get("FLUX_MCP_EVAL_MODEL", "claude-sonnet-5-5"))
    assert summary(results)["passed"] >= 1
