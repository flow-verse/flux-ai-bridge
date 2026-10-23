# SPDX-License-Identifier: GPL-3.0-or-later
"""A stand-in for ``flux-engine serve`` used to test supervision without the real engine.

Methods: ping -> pong; hang -> never answers; crash -> exits 3 without answering;
noise -> writes a non-JSON line, then answers; anything else -> -32601.
"""
import json
import sys

print(json.dumps({"jsonrpc": "2.0", "method": "ready", "params": {"api_version": "0.1", "sample_rate": 48000}}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    m, rid = req.get("method"), req.get("id")
    if m == "hang":
        continue
    if m == "crash":
        print("fake engine: crashing", file=sys.stderr, flush=True)
        sys.exit(3)
    if m == "noise":
        print("this is not JSON", flush=True)
    if m in ("ping", "noise"):
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"pong": True}}), flush=True)
    elif m == "shutdown":
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"ok": True}}), flush=True)
        break
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method '{m}'"}}), flush=True)
