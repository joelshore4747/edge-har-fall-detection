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

# Resolve credentials. ADMIN_USERNAME / ADMIN_PASSWORD come from .env (matches
# .env.compose.example). UNIFALL_USERNAME / UNIFALL_PASSWORD are accepted as
# overrides for ad-hoc runs.
USERNAME="${UNIFALL_USERNAME:-${ADMIN_USERNAME:-}}"
PASSWORD="${UNIFALL_PASSWORD:-${ADMIN_PASSWORD:-}}"

if [[ -z "$USERNAME" || -z "$PASSWORD" ]]; then
  echo "ERROR: ADMIN_USERNAME / ADMIN_PASSWORD (or UNIFALL_USERNAME / UNIFALL_PASSWORD) must be set." >&2
  echo "       Add them to $ROOT_DIR/.env or export them in the shell before running." >&2
  exit 1
fi

BASE_URL="${UNIFALL_BASE_URL:-https://api.unifallmonitor.com}"
CACHE_DIR="${UNIFALL_CACHE_DIR:-$ROOT_DIR/artifacts/unifallmonitor/cache}"
# OUT_DIR is only set when the caller explicitly overrides it. Leaving it
# unset lets the trainer mint a run-id-scoped directory under
# UNIFALL_RUNS_ROOT, which is what `_train_recordings --mark-current`
# requires (the symlink in `artifacts/unifallmonitor/current` points at
# that run-id directory).
OUT_DIR="${UNIFALL_OUT_DIR:-}"

ARGS=(
  "$ROOT_DIR/scripts/train_unifallmonitor_har.py"
  --base-url "$BASE_URL"
  --username "$USERNAME"
  --password "$PASSWORD"
  --cache-dir "$CACHE_DIR"
  --per-placement
)
if [[ -n "$OUT_DIR" ]]; then
  ARGS+=(--out "$OUT_DIR")
fi

# Forward any extra flags (e.g. --max-sessions 50, --dry-run, --refresh-cache).
ARGS+=("$@")

echo "[run_train_unifallmonitor_har] base_url=$BASE_URL"
echo "[run_train_unifallmonitor_har] username=$USERNAME"
echo "[run_train_unifallmonitor_har] cache_dir=$CACHE_DIR"
echo "[run_train_unifallmonitor_har] out_dir=${OUT_DIR:-<run-id-scoped under runs/>}"

exec "$PYTHON_BIN" "${ARGS[@]}"
