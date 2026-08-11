#!/usr/bin/env bash
# Pipeline completo: SQL + CodeXGLUE → SFT → LoRA → eval → export.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${1:-configs/train_full.yaml}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Crie o .venv e instale deps antes."
  exit 1
fi

export PYTHONPATH="$ROOT"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

mkdir -p outputs/training_full
LOG="outputs/training_full/train.log"

echo "==> 1) SQL (Spider)"
"$PYTHON" scripts/prepare_sql_spider.py --config "$CONFIG"

echo "==> 2) CodeXGLUE + merge SQL"
"$PYTHON" scripts/prepare_dataset.py --config "$CONFIG"

echo "==> 3) SFT"
"$PYTHON" scripts/build_sft_from_codexglue.py --config "$CONFIG"

BACKEND=cpu
if "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  BACKEND=cuda
elif "$PYTHON" -c "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)"; then
  BACKEND=mps
fi
echo "==> 4) Treino LoRA (backend=$BACKEND)"
"$PYTHON" scripts/train_lora.py \
  --config "$CONFIG" \
  --backend "$BACKEND" \
  --resume \
  2>&1 | tee -a "$LOG"

ADAPTER="$( "$PYTHON" -c "from legacy_doc.config import load_config; print(load_config('$CONFIG')['project']['output_dir'])" )"

echo "==> 5) Avaliar"
"$PYTHON" scripts/evaluate_model.py --config "$CONFIG" --adapter "$ADAPTER"

EXPORT="models/export/qwen2.5-1.5b-celx"
echo "==> 6) Exportar → $EXPORT"
"$PYTHON" scripts/export_model.py --adapter "$ADAPTER" --output "$EXPORT"

# Inferência smoke: PHP se o config não tiver Python.
SMOKE_FILE="dataset/examples/calcula_total.php"
SMOKE_LANG="php"
if "$PYTHON" -c "from legacy_doc.config import load_config; c=load_config('$CONFIG'); raise SystemExit(0 if 'python' in [str(x).lower() for x in c['data'].get('target_languages',[])] else 1)"; then
  SMOKE_FILE="dataset/examples/calcula_total.py"
  SMOKE_LANG="python"
fi
echo "==> 7) Smoke inferência ($SMOKE_LANG)"
"$PYTHON" scripts/document_code.py \
  --file "$SMOKE_FILE" \
  --language "$SMOKE_LANG" \
  --adapter "$EXPORT/adapter" \
  --config "$CONFIG" \
  | head -n 60

echo "Pronto: $EXPORT"
