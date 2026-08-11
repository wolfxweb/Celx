# Notebooks

| Notebook | Quando | Dados |
|---|---|---|
| `01_benchmark.ipynb` | Escolher modelo | 4 casos locais |
| `02_treino_qlora.ipynb` | Smoke / validar pipeline | dataset mínimo |
| `03_treino_real.ipynb` | Treino real + export | CodeXGLUE → LoRA |
| `04_pipeline_completo.ipynb` | **Produção end-to-end** | Stage A/B/C — Mac ou **RunPod Jupyter** |

```text
01_benchmark → 04_pipeline_completo  (recomendado; também no RunPod)
            ↘ 03_treino_real         (só CodeXGLUE, sem SQL)
            ↘ 02_treino_qlora        (smoke)
```

Kernel local: **Python (Celx .venv)**  
RunPod: template Jupyter + `docs/RUNPOD.md`  
Watch: `python scripts/watch_training.py`  

| Ambiente | Config stage A |
|---|---|
| M1 local | `configs/train_full.yaml` |
| CUDA / A4500 (RunPod) | `configs/train_php_js_sql_full.yaml` |
| Bilingue | `configs/train_bilingual.yaml` |

```bash
bash scripts/package_project.sh   # → dist/Celx-colab.zip (upload no RunPod)
bash scripts/pipeline_full.sh configs/train_php_js_sql_full.yaml
```
