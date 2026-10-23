# SPDX-License-Identifier: GPL-3.0-or-later
"""flux-mcp: a Model Context Protocol server for Flux.

It owns one ``flux-engine serve`` child (JSON-RPC 2.0 over stdio) and one
``flux-ai serve`` warm worker (flux-ai job protocol v2), and exposes a small,
curated tool set to MCP clients over stdio only. See docs/mcp.md.
"""

__version__ = "0.1.0"
