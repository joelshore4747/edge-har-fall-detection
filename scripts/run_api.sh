#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR"
fi

cd "$ROOT_DIR"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
API_WORKERS="${API_WORKERS:-1}"

case "$API_WORKERS" in
  ''|*[!0-9]*)
    echo "ERROR: API_WORKERS must be a positive integer." >&2
    exit 1
    ;;
esac

if [[ "$API_WORKERS" -lt 1 ]]; then
  echo "ERROR: API_WORKERS must be at least 1." >&2
  exit 1
fi

UVICORN_ARGS=(
  -m uvicorn
  apps.api.main:app
  --host "${API_HOST:-0.0.0.0}"
  --port "${API_PORT:-8000}"
  --workers "$API_WORKERS"
  --log-level "$(printf '%s' "${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')"
)

case "${UVICORN_ACCESS_LOG:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    ;;
  *)
    UVICORN_ARGS+=(--no-access-log)
    ;;
esac

exec "$PYTHON_BIN" "${UVICORN_ARGS[@]}"
