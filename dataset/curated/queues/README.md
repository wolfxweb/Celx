# Filas de curadoria pt-BR

Código + referência EN; `documentation_pt` vazio até revisão humana.

| Modo | Total | Por lang | Comando |
|---|---:|---:|---|
| piloto | 280 | 70 | `... --mode piloto` |
| meta_v1 | 2.400 | 600 | `... --mode meta_v1` |
| **escala_2k (atual)** | **8.000** | **2.000** | `python scripts/create_curated_queue.py --config configs/curadoria.yaml --mode escala_2k` |

Split por linguagem: **1600 train + 200 val + 200 test**. Linguagens: python / php / javascript / sql.

Depois de aprovar, copie/mova registros com `review_status=approved` para `../{train,validation,test}.jsonl`.
