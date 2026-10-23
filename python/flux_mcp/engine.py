# SPDX-License-Identifier: GPL-3.0-or-later
"""flux-engine client for flux-mcp. The implementation lives in :mod:`flux_ai.engine_client`
so the long-form renderer can share it; this module re-exports it."""
from __future__ import annotations

import sys

from flux_ai.engine_client import (INVALID_PARAMS, NO_PROJECT, OP_FAILED, STATE_CONFLICT, UNKNOWN_ID,  # noqa: F401
                                   WRONG_TYPE, EngineClient, EngineCrashed, EngineError, EngineNotFound,
                                   find_engine)


def log(msg: str) -> None:
    """Diagnostics go to stderr; stdout belongs to the MCP protocol."""
    print(f"flux-mcp: {msg}", file=sys.stderr, flush=True)
