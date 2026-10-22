#!/usr/bin/env bash
# End-to-end CLAP host test: bundle the plugin, then drive it with tests/host/capture_host.c.
# Needs: Rust, clang, a working flux-ai (default: .venv/bin/flux-ai), git (to fetch CLAP headers).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-$ROOT/target/host-test}"
FLUX_AI_BIN="${FLUX_AI_BIN:-$ROOT/.venv/bin/flux-ai}"
SECS="${1:-4}"

mkdir -p "$WORK"
[ -d "$WORK/clap" ] || git clone -q --depth 1 https://github.com/free-audio/clap.git "$WORK/clap"
(cd "$ROOT" && cargo xtask bundle flux-ai-clap --release)
clang -O1 -Wall -I"$WORK/clap/include" "$ROOT/tests/host/capture_host.c" -o "$WORK/capture_host"
rm -rf "$WORK/out" && mkdir -p "$WORK/out"
FLUX_AI_BIN="$FLUX_AI_BIN" FLUX_AI_OUT="$WORK/out" \
  "$WORK/capture_host" "$ROOT/target/bundled/Flux AI Bridge.clap" "$WORK/out" "$SECS"
