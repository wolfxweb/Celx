#!/usr/bin/env bash
# Sobe o adapter Celx como API OpenAI em http://127.0.0.1:8000/v1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
exec "$ROOT/.venv/bin/python" scripts/serve_openai_adapter.py "$@"
