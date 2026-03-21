#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

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

echo "Starting frontend on http://127.0.0.1:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"
) &
frontend_pid=$!

echo

echo "Open http://127.0.0.1:$FRONTEND_PORT"

echo "Press Ctrl+C to stop services started by this script."

echo

wait
