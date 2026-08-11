# Execução no Kaggle

Use um Kaggle Notebook comum em `https://www.kaggle.com/code/new`.
Não use a área Kaggle Benchmarks / Benchmark Task.

## Benchmark (Fase 2)

### Local (smoke)

Abra `notebooks/01_benchmark.ipynb` neste Mac e execute as células.
Usa `configs/benchmark_local.yaml` + 4 casos de `baseline.jsonl` (CPU possível, lento).

### Kaggle (400 casos)

1. Neste Mac: `bash scripts/package_project.sh` → `dist/Celx-colab.zip`
2. Abra `notebooks/01_benchmark.ipynb` no Kaggle (upload do `.ipynb`).
3. Settings → GPU T4 ou P100 + Internet ligada.
4. Add Input → dataset com `Celx-colab.zip`.
5. Execute as células (um modelo por sessão).
6. Baixe `celx-benchmark-results.zip` e copie para `outputs/benchmark_100/`.

Para continuar outra sessão, adicione o JSONL parcial como Input; o executor ignora IDs já concluídos.

## Treino (Fase 4)

Treino QLoRA / pipeline completo: preferencialmente **RunPod + Jupyter** (`docs/RUNPOD.md`,
notebook `04_pipeline_completo.ipynb`).
No Kaggle só faça se houver tempo/GPU suficientes e o dataset SFT já estiver no ZIP.
