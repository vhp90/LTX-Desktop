#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export LTX_BROWSER_ONLY=1
export LTX_APP_DATA_DIR="${LTX_APP_DATA_DIR:-$PROJECT_DIR/.ltx-data}"
export LTX_BACKEND_BIND_HOST="${LTX_BACKEND_BIND_HOST:-0.0.0.0}"
export LTX_PORT="${LTX_PORT:-18000}"
export LTX_ALLOWED_ORIGIN_REGEX="${LTX_ALLOWED_ORIGIN_REGEX:-^https://(lightning\\.ai|.*\\.cloudspaces\\.litng\\.ai)$}"

cd "$PROJECT_DIR"
corepack pnpm exec concurrently -k \
  "cd backend && uv run --python 3.13.7 python ltx2_server.py" \
  "vite"
