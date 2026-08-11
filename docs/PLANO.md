# Plano do ciclo atual

Projeto reiniciado. Snapshot anterior em `arquivos/`.

## Notebooks ativos

| # | Notebook | Fase |
|---|---|---|
| 1 | `notebooks/01_benchmark.ipynb` | Seleção do modelo (local) |
| 2 | `notebooks/02_treino_qlora.ipynb` | Smoke / validar pipeline |
| 3 | `notebooks/03_treino_real.ipynb` | Treino real CodeXGLUE → LoRA |
| 4 | `notebooks/04_pipeline_completo.ipynb` | **End-to-end:** dados → treino → eval → export |

## Fases

| Fase | Objetivo | Critério de aceite |
|---|---|---|
| 2 — Benchmark | Escolher modelo-base | Decisão em `docs/DECISOES.md` |
| 3 — Dados e SFT | CodeXGLUE (+ SQL) → SFT | ≥100 exemplos train |
| 4 — Treino / eval / export | LoRA + avaliação estrutural + pacote | `models/export/` |

## Pipeline completo (recomendado)

Abra `notebooks/04_pipeline_completo.ipynb` com kernel `.venv`, ou:

```bash
source .venv/bin/activate
bash scripts/pipeline_full.sh
```

Fluxo:

1. `prepare_sql_spider.py` → `dataset/sql/`
2. `prepare_dataset.py` → CodeXGLUE + merge SQL → `dataset/processed/full/`
3. `build_sft_from_codexglue.py` → `dataset/processed/sft_full/`
4. `train_lora.py` → `models/qwen3-legacy-doc-lora-full/`
5. `evaluate_model.py` → `outputs/eval_full/`
6. `export_model.py` → `models/export/qwen2.5-1.5b-celx/` + Ollama `qwen2.5:1.5b-celx`

Config: `configs/train_full.yaml` (M1) ou `configs/train_php_js_sql_full.yaml` (CUDA: PHP/JS/SQL cheios, sem Python).  
Sem fila curada no stage A; stage B/C no notebook cobrem bilingue + inferência.

| Artefato | Uso |
|---|---|
| `dataset/processed/full_php_js_sql/` ou `full/` | normalizado |
| `dataset/processed/sft_php_js_sql/` ou `sft_full/` | SFT stage A |
| `dataset/processed/sft_bilingual/` | SFT pt-BR + en |
| `outputs/training_php_js_sql/` | live.json, metrics, STATUS |
| `models/export/qwen2.5-1.5b-celx/` | pacote + Ollama |

No M1: ~400 exemplos/linguagem. Na A4500: base cheia PHP+JS+SQL (~300k) com QLoRA.

## Treino cheio PHP + JS + SQL (sem Python)

Na A4500 (20 GB), use a base CodeXGLUE/Spider **inteira** só dessas linguagens:

| Linguagem | Train (approx) |
|---|---:|
| PHP | ~241.241 |
| JavaScript | ~58.025 |
| SQL (Spider) | ~5–7k |
| **Total** | **~300k+** (não ~500k — isso incluía Python) |

```bash
bash scripts/pipeline_full.sh configs/train_php_js_sql_full.yaml
# ou QLoRA direto:
python scripts/train_qlora.py --config configs/train_php_js_sql_full.yaml
```

Custo esperado nessa pod (~\$0,25/h): ordem de **\$5–15** (1 época), conforme `max_length`/throughput. Disco 40 GB: baixe só php+js+spider e `save_total_limit: 1`.

## Multi-idioma (pt-BR + en)

Dois alvos por código: `documentation_pt` + `documentation_en`.

```bash
python scripts/build_sft_bilingual.py --config configs/train_bilingual.yaml
python scripts/document_code.py --file dataset/examples/calcula_total.php --language php --doc-language pt-BR
python scripts/document_code.py --file dataset/examples/calcula_total.php --language php --doc-language en
```

## Base pt-BR (curadoria)

Inventário e decisão local vs nuvem:

```bash
python scripts/inventory_datasets.py   # → dataset/curated/INVENTARIO.md
```

| Volume pt-BR | Onde |
|---:|---|
| Piloto 280 | M1 |
| Meta 2.400 | M1 ou nuvem |
| 8.000 (2k/lang) | Nuvem |

Fila: `configs/curadoria.yaml` + `scripts/create_curated_queue.py`. Smoke fica em `dataset/curated_smoke/`.

## Referências

- Escopo: `docs/ESCOPO.md`
- Seleção: `docs/SELECAO_MODELO.md`
- Rubrica: `docs/RUBRICA.md`
- Histórico: `arquivos/`
