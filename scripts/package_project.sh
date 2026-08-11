#!/usr/bin/env bash
# Empacota a raiz ativa para Colab/Kaggle/RunPod Jupyter (exclui arquivos/ e artefatos locais).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/Celx-colab.zip}"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

cd "$ROOT"
tar -a -c -f "$OUT" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.venv' \
  --exclude='arquivos' \
  --exclude='.git' \
  --exclude='models/*' \
  --exclude='outputs/*' \
  --exclude='dataset/raw' \
  --exclude='dataset/processed' \
  --exclude='.hf_cache' \
  --exclude='dist' \
  .gitignore \
  README.md \
  pyproject.toml \
  requirements.txt \
  configs \
  dataset/benchmark \
  dataset/curated \
  dataset/examples \
  dataset/sql \
  docs \
  legacy_doc \
  notebooks \
  scripts \
  tests \
  models/.gitkeep \
  outputs/.gitkeep

echo "$OUT ($(du -h "$OUT" | awk '{print $1}'))"
echo "RunPod: envie para /workspace/Celx-colab.zip e abra notebooks/04_pipeline_completo.ipynb"
echo "Guia: docs/RUNPOD.md"
