#!/usr/bin/env bash
#
# Launch the Graph of Trace MCP server and the frontend viewer together.
#
# The MCP server is configured (via env) to write got.json into
# frontend/public/, which the viewer serves and polls — so the UI shows the
# trace live as an agent records subtasks.
#
# Usage:
#   scripts/start.sh
#   GOT_TRANSPORT=sse GOT_PORT=9000 GOT_UI_PORT=4500 scripts/start.sh
#
# Env overrides:
#   GOT_TRANSPORT   MCP transport: stdio | sse | streamable-http   (default: streamable-http)
#   GOT_HOST        MCP bind host                                   (default: 127.0.0.1)
#   GOT_PORT        MCP bind port                                   (default: 8000)
#   GOT_UI_PORT     Frontend dev server port                       (default: 4500)
#   PYTHON          Python interpreter                              (default: python3)
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRANSPORT="${GOT_TRANSPORT:-streamable-http}"
MCP_HOST="${GOT_HOST:-127.0.0.1}"
MCP_PORT="${GOT_PORT:-8000}"
UI_PORT="${GOT_UI_PORT:-4500}"
PYTHON="${PYTHON:-python3}"

# Make the backend write got.json where the frontend serves it.
export GOT_OUTPUT_BASE_DIR="${GOT_OUTPUT_BASE_DIR:-$REPO_DIR/frontend/public}"
export GOT_OUTPUT_PATH_TEMPLATE="${GOT_OUTPUT_PATH_TEMPLATE:-{base_dir}/got.json}"

echo "[start] repo:        $REPO_DIR"
echo "[start] MCP server:  $TRANSPORT  $MCP_HOST:$MCP_PORT"
echo "[start] UI viewer:   http://localhost:$UI_PORT"
echo "[start] got.json ->  $GOT_OUTPUT_BASE_DIR/got.json"

# Install frontend deps on first run.
if [ ! -d "$REPO_DIR/frontend/node_modules" ]; then
  echo "[start] installing frontend deps (first run)..."
  (cd "$REPO_DIR/frontend" && npm install)
fi

pids=()
cleanup() {
  echo
  echo "[start] shutting down..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# MCP server (writes got.json to frontend/public/).
( cd "$REPO_DIR" && exec "$PYTHON" server.py \
    --transport "$TRANSPORT" --host "$MCP_HOST" --port "$MCP_PORT" ) &
pids+=("$!")

# Frontend dev server.
( cd "$REPO_DIR/frontend" && exec npm run dev -- --port "$UI_PORT" ) &
pids+=("$!")

echo "[start] both running. Press Ctrl+C to stop."

# Exit (and tear down the other process) as soon as either one exits.
wait -n
