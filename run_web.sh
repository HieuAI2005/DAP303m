#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Use Node v20+ from nvm if system node is too old
NODE_BIN="node"
VITE_BIN="$FRONTEND_DIR/node_modules/.bin/vite"
for candidate in \
    "$HOME/.nvm/versions/node/v20.19.6/bin/node" \
    "$HOME/.nvm/versions/node/v24.14.0/bin/node" \
    /usr/local/bin/node; do
  if [[ -x "$candidate" ]]; then
    _ver=$("$candidate" -e 'process.stdout.write(process.versions.node.split(".")[0])' 2>/dev/null || echo 0)
    if (( _ver >= 18 )); then
      NODE_BIN="$candidate"
      break
    fi
  fi
done

backend_pid=""
frontend_pid=""

cleanup() {
  if [[ -n "${frontend_pid}" ]] && kill -0 "${frontend_pid}" 2>/dev/null; then
    kill "${frontend_pid}" 2>/dev/null || true
  fi
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

port_in_use() {
  local port="$1"
  ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .
}

echo "Project root: $ROOT_DIR"

if port_in_use "$BACKEND_PORT"; then
  echo "Backend port $BACKEND_PORT is already in use. Reusing existing backend."
else
  echo "Starting backend on http://127.0.0.1:$BACKEND_PORT"
  (
    cd "$ROOT_DIR"
    PYTHONPATH=src python -m uvicorn movierag.web_api:app --host 127.0.0.1 --port "$BACKEND_PORT"
  ) &
  backend_pid=$!
fi

if port_in_use "$FRONTEND_PORT"; then
  echo "Frontend port $FRONTEND_PORT is already in use."
  echo "Open http://127.0.0.1:$FRONTEND_PORT"
  wait
  exit 0
fi

echo "Starting frontend on http://127.0.0.1:$FRONTEND_PORT  (node: $NODE_BIN)"
(
  cd "$FRONTEND_DIR"
  "$NODE_BIN" "$VITE_BIN" dev --host 127.0.0.1 --port "$FRONTEND_PORT"
) &
frontend_pid=$!

echo

echo "Open http://127.0.0.1:$FRONTEND_PORT"

echo "Press Ctrl+C to stop services started by this script."

echo

wait
