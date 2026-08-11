# Inventário de dados Celx

Gerado por `python scripts/inventory_datasets.py`.

pt-BR real = documentation_pt curada/aprovada. EN embrulhado = reference CodeXGLUE/Spider em template PT (não é documentação pt-BR).

## Metas pt-BR

| Cenário | Train/lang | Val/lang | Test/lang | Total (4 langs) | Onde treinar |
|---|---:|---:|---:|---:|---|
| Smoke | ~4 | ~1 | ~1 | ~19 | M1 (pipeline) |
| Piloto | 50 | 10 | 10 | **280** | M1 |
| Meta v1 | 500 | 50 | 50 | **2.400** | M1 ou nuvem |
| Escala 2k | 1600 | 200 | 200 | **8.000** (2k/lang) | Nuvem QLoRA |

## Onde treinar (orientação)

| Volume pt-BR aprovado | Onde | Motivo |
|---:|---|---|
| ≤ 50 (smoke) | M1 local | Só valida pipeline |
| 200–800 (piloto) | M1 local | Cabe em LoRA/MPS; qualidade real começa aqui |
| 800–2.400 (meta v1) | M1 ou nuvem | M1 lento; RunPod/Kaggle se quiser 1 época rápida |
| 8.000 (escala 2k/lang) | Nuvem QLoRA | Fora do conforto do M1 8GB |

- **pt-BR aprovado agora:** 5
- **fila pendente (a curar):** 8000
- **EN embrulhado (CodeXGLUE local, se existir):** 6400

**Decisão sugerida:** continue no M1 com o notebook 04 (EN embrulhado) só para pipeline; em paralelo cure o piloto pt-BR (50/10/10 por linguagem).

## pt-BR aprovado (`dataset/curated`)

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/curated`
- Total: **5**
- Com `documentation_pt`: **5**
- Linguagens: `{'javascript': 1, 'php': 1, 'python': 2, 'sql': 1}`
- Fontes: `{'exemplo_manual': 5}`
- Status: `{'approved': 5}`

- `train`: 4, langs={'javascript': 1, 'php': 1, 'python': 1, 'sql': 1}
- `validation`: 1, langs={'python': 1}
- `test`: 0, langs={}

## Smoke pt-BR (`dataset/curated_smoke`)

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/curated_smoke`
- Total: **15**
- Com `documentation_pt`: **15**
- Linguagens: `{'javascript': 4, 'php': 4, 'python': 4, 'sql': 3}`
- Fontes: `{'smoke_local': 15}`
- Status: `{'approved': 15}`

- `train`: 12, langs={'javascript': 3, 'php': 3, 'python': 3, 'sql': 3}
- `validation`: 1, langs={'python': 1}
- `test`: 2, langs={'javascript': 1, 'php': 1}

## Filas de curadoria (`dataset/curated/queues`)

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/curated/queues`
- Total: **8000**
- Pendentes: **8000**
- Linguagens: `{'javascript': 2000, 'php': 2000, 'python': 2000, 'sql': 2000}`

- `train`: 6400, pending=6400, langs={'javascript': 1600, 'php': 1600, 'python': 1600, 'sql': 1600}
- `validation`: 800, pending=800, langs={'javascript': 200, 'php': 200, 'python': 200, 'sql': 200}
- `test`: 800, pending=800, langs={'javascript': 200, 'php': 200, 'python': 200, 'sql': 200}

## Normalizado EN (`dataset/processed/codexglue`)

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/processed/codexglue`
- Total: **1200**
- Linguagens: `{'javascript': 400, 'php': 400, 'python': 400}`

- `train`: 900, langs={'javascript': 300, 'php': 300, 'python': 300}
- `validation`: 150, langs={'javascript': 50, 'php': 50, 'python': 50}
- `test`: 150, langs={'javascript': 50, 'php': 50, 'python': 50}

## Normalizado EN (`dataset/processed/full`)

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/processed/full`
- Total: **8000**
- Linguagens: `{'javascript': 2000, 'php': 2000, 'python': 2000, 'sql': 2000}`

- `train`: 6400, langs={'javascript': 1600, 'php': 1600, 'python': 1600, 'sql': 1600}
- `validation`: 800, langs={'javascript': 200, 'php': 200, 'python': 200, 'sql': 200}
- `test`: 800, langs={'javascript': 200, 'php': 200, 'python': 200, 'sql': 200}

## SFT (`dataset/processed/sft`) — kind=`?`

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/processed/sft`
- Total: **5**

- `train`: 1, langs={'sql': 1}
- `validation`: 2, langs={'python': 2}
- `test`: 2, langs={'javascript': 1, 'php': 1}

## SFT (`dataset/processed/sft_real`) — kind=`real_stage1`

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/processed/sft_real`
- Total: **1200**

- `train`: 900, langs={'javascript': 300, 'php': 300, 'python': 300}
- `validation`: 150, langs={'javascript': 50, 'php': 50, 'python': 50}
- `test`: 150, langs={'javascript': 50, 'php': 50, 'python': 50}

## SFT (`dataset/processed/sft_full`)

- Path: `/Users/wolfx/Documents/Dev/Celx/dataset/processed/sft_full`
- Status: **ausente**
