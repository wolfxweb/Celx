# Curadoria em português (pt-BR)

Esta pasta é a **base real** de documentação em português.

## Estrutura

| Path | Papel |
|---|---|
| `train.jsonl` / `validation.jsonl` / `test.jsonl` | Exemplos **aprovados** (`documentation_pt` + `review_status=approved`) |
| `queues/*_review_queue.jsonl` | Fila humana (código + referência EN; `documentation_pt` vazio) |
| `templates/` | Contrato e exemplo |
| `INVENTARIO.md` | Contagem atual (gerar com o script abaixo) |
| `../curated_smoke/` | Smoke do pipeline — **não** misturar com curadoria real |

## Formato aprovado

```json
{"id":"...","language":"php","code":"...","documentation_pt":"## Método ou função\n...","documentation_en":"## Method or function\n...","source":"code_x_glue","review_status":"approved"}
```

`documentation_pt` (pt-BR) e, para multi-idioma, `documentation_en` (en) precisam das seções do contrato no respectivo idioma.
Docstrings EN do CodeXGLUE **não** entram em `documentation_pt` sem tradução/reestruturação/revisão.

SFT bilingue: `python scripts/build_sft_bilingual.py --config configs/train_bilingual.yaml`

## Linguagens obrigatórias

O treino Celx cobre **as quatro** linguagens, com o mesmo volume por idioma:

- Python
- PHP
- JavaScript
- SQL

A fila piloto já nasce balanceada (70 por linguagem). A semente aprovada em `train.jsonl` traz 1 exemplo de cada.

## Volumes (para decidir onde treinar)

| Cenário | Total | Por linguagem | Onde treinar |
|---|---:|---:|---|
| Smoke | ~19 | misto | M1 (só pipeline) |
| Piloto | 280 | 70 (50/10/10) | M1 |
| Meta v1 | 2.400 | 600 (500/50/50) | M1 ou nuvem |
| **Escala 2k (pt-BR)** | **8.000** | **2.000** (1600/200/200) | **Nuvem QLoRA** |
| EN cheio PHP+JS+SQL | ~300k train | tudo CodeXGLUE/Spider | Nuvem A4500 (`train_php_js_sql_full.yaml`) |

> ~500k incluía Python. Sem Python: **~241k PHP + ~58k JS + ~7k SQL**.


## Comandos

```bash
# Inventário (pt-BR vs EN embrulhado)
python scripts/inventory_datasets.py

# Pool EN + fila piloto (280 pendentes)
python scripts/prepare_sql_spider.py --config configs/curadoria.yaml
python scripts/prepare_dataset.py --config configs/curadoria.yaml
python scripts/create_curated_queue.py --config configs/curadoria.yaml --mode piloto

# Meta v1 (2.400) — só depois do piloto
python scripts/create_curated_queue.py --config configs/curadoria.yaml --mode meta_v1
```

Checklist: `docs/CHECKLIST_CURADORIA.md`. Fluxo: `docs/CURADORIA_SFT.md`.
