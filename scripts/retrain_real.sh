#!/usr/bin/env bash
# Retreina o modelo real (CodeXGLUE) e exporta o adapter para uso.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Crie o .venv e instale deps antes."
  exit 1
fi

export PYTHONPATH="$ROOT"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

mkdir -p outputs/training_real
LOG="outputs/training_real/train.log"

echo "==> Treino real (log: $LOG)"
"$PYTHON" scripts/train_lora.py \
  --config configs/train_real.yaml \
  --backend mps \
  --output-dir models/qwen3-legacy-doc-lora-real \
  --resume \
  2>&1 | tee -a "$LOG"

echo "==> Exportar adapter"
"$PYTHON" scripts/export_model.py \
  --adapter models/qwen3-legacy-doc-lora-real \
  --output models/export/qwen2.5-1.5b-celx

echo "==> Teste rápido de inferência"
"$PYTHON" scripts/document_code.py \
  --file dataset/examples/calcula_total.py \
  --language python \
  --adapter models/export/qwen2.5-1.5b-celx/adapter \
  --config configs/train_real.yaml \
  | head -n 40

echo "Pronto: models/export/qwen2.5-1.5b-celx/ (Ollama: qwen2.5:1.5b-celx)"
