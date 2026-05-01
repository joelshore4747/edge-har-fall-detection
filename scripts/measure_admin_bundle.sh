#!/usr/bin/env bash
set -euo pipefail

# Build the admin app and print a one-screen size report for every chunk in
# `apps/admin/dist/assets/`. Reports raw and gzipped sizes so the user can
# paste before/after diffs into commits or PR descriptions.
#
# Usage:
#   bash scripts/measure_admin_bundle.sh           # standard build
#   ANALYZE=true bash scripts/measure_admin_bundle.sh   # also emits dist/stats.html

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADMIN_DIR="$ROOT_DIR/apps/admin"

if [[ ! -d "$ADMIN_DIR" ]]; then
  echo "ERROR: $ADMIN_DIR not found." >&2
  exit 1
fi

cd "$ADMIN_DIR"

NPM_BIN="${NPM_BIN:-npm}"

if [[ ! -d "$ADMIN_DIR/node_modules" ]]; then
  echo "+ $NPM_BIN ci  (node_modules missing)"
  "$NPM_BIN" ci
fi

echo "+ $NPM_BIN run build"
if [[ "${ANALYZE:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  ANALYZE=true "$NPM_BIN" run build
else
  "$NPM_BIN" run build
fi

ASSET_DIR="$ADMIN_DIR/dist/assets"
if [[ ! -d "$ASSET_DIR" ]]; then
  echo "ERROR: $ASSET_DIR was not produced by the build." >&2
  exit 1
fi

printf "\n%-46s  %10s  %10s  %6s\n" "asset" "raw bytes" "gzip bytes" "ratio"
printf "%-46s  %10s  %10s  %6s\n" "$(printf '%0.s=' {1..46})" "==========" "==========" "======"

shopt -s nullglob
TOTAL_RAW=0
TOTAL_GZIP=0
for asset in "$ASSET_DIR"/*; do
  [[ -f "$asset" ]] || continue
  name="$(basename "$asset")"
  raw=$(stat -c "%s" "$asset")
  gzipped=$(gzip -c -- "$asset" | wc -c)
  if (( raw > 0 )); then
    ratio=$(awk -v r="$raw" -v g="$gzipped" 'BEGIN { printf "%.0f%%", (g / r) * 100 }')
  else
    ratio="-"
  fi
  printf "%-46s  %10d  %10d  %6s\n" "$name" "$raw" "$gzipped" "$ratio"
  TOTAL_RAW=$(( TOTAL_RAW + raw ))
  TOTAL_GZIP=$(( TOTAL_GZIP + gzipped ))
done
printf "%-46s  %10s  %10s\n" "$(printf '%0.s-' {1..46})" "----------" "----------"
printf "%-46s  %10d  %10d\n" "TOTAL" "$TOTAL_RAW" "$TOTAL_GZIP"

if [[ -f "$ADMIN_DIR/dist/stats.html" ]]; then
  echo
  echo "Bundle visualiser written to apps/admin/dist/stats.html"
fi
