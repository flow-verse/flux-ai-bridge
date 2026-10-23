# SPDX-License-Identifier: GPL-3.0-or-later
"""DAWZY-style task suite for flux-mcp.

Each task has a natural-language prompt, a setup (run through the MCP tools),
a reference tool-call sequence and a programmatic checker over the engine's
``describe_project`` and the files on disk. Two runners share it:

- scripted (``run_scripted``): executes the reference sequence. Deterministic;
  runs in pytest and CI and proves every task is solvable with the tool set.
- LLM (``run_llm``): a model drives the MCP tools from the prompt alone. Needs
  ``ANTHROPIC_API_KEY`` (or another Anthropic credential); skipped otherwise.

Usage: ``python -m flux_mcp.evals [--llm] [--model ID] [--task ID ...] [--json out.json]``
"""
