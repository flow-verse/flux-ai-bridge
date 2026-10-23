# SPDX-License-Identifier: GPL-3.0-or-later
"""``flux-mcp``: run the Flux MCP server over stdio.

stdio is the only transport: the server is reachable only by the process that
started it (the report's localhost-only decision; no network listener, so no
AGPL network-use exposure of flux-engine). Diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flux_mcp import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flux-mcp", description="Flux MCP server (stdio)")
    p.add_argument("--engine", help="path to flux-engine (default: $FLUX_ENGINE, then PATH)")
    p.add_argument("--plugin-path", action="append", default=[], metavar="DIR",
                   help="extra CLAP plugin folder passed to flux-engine (repeatable)")
    p.add_argument("--vst3-path", action="append", default=[], metavar="DIR", help="extra VST3 folder (repeatable)")
    p.add_argument("--sr", type=int, help="engine sample rate (default 48000)")
    p.add_argument("--flux-ai", help="command that runs flux-ai (default: this Python's flux_ai, then PATH)")
    p.add_argument("--root", type=Path, help="only allow file paths inside this directory")
    p.add_argument("--workdir", type=Path, help="base for relative paths (default: current directory)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-call engine timeout in seconds (default 60)")
    p.add_argument("--render-timeout", type=float, default=1800.0, help="render timeout in seconds (default 1800)")
    p.add_argument("--version", action="version", version=f"flux-mcp {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        from flux_mcp.server import Config, Session, build_server
    except ImportError as e:  # the [mcp] extra is missing
        print(f"flux-mcp: {e}. Install with: pip install 'flux-ai[mcp]'", file=sys.stderr)
        return 2
    engine_args: list[str] = []
    for d in a.plugin_path:
        engine_args += ["--plugin-path", str(Path(d).expanduser())]
    for d in a.vst3_path:
        engine_args += ["--vst3-path", str(Path(d).expanduser())]
    if a.sr:
        engine_args += ["--sr", str(a.sr)]
    cfg = Config(engine=a.engine, engine_args=engine_args, flux_ai=a.flux_ai.split() if a.flux_ai else None,
                 root=a.root.expanduser() if a.root else None,
                 workdir=(a.workdir or Path(os.getcwd())).expanduser(),
                 call_timeout=a.timeout, render_timeout=a.render_timeout)
    server = build_server(Session(cfg))
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
