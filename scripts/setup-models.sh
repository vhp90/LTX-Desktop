#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export LTX_HF_USE_XET="${LTX_HF_USE_XET:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export LTX_HF_DOWNLOAD_RETRIES="${LTX_HF_DOWNLOAD_RETRIES:-6}"

cd "$PROJECT_DIR/backend"
uv run --python 3.13.7 python ../scripts/setup-models.py "$@"
