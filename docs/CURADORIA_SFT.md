# Curadoria do dataset SFT

## Objetivo

Produzir exemplos supervisionados em português que ensinem o Qwen3-1.7B a documentar somente
comportamentos sustentados pelo código. O benchmark de 400 casos permanece reservado para
avaliação e é excluído por hash das filas e do dataset final.

## Meta inicial

| Divisão | Por linguagem | Total |
|---|---:|---:|
| Treino | 500 | 2.000 |
| Validação | 50 | 200 |
| Teste curado | 50 | 200 |
| **Total** | **600** | **2.400** |

Para um ensaio barato, comece com **50/10/10 por linguagem (280 total)** — cabe no M1.
A meta de 2.400 é a primeira versão curada; a escala atual do projeto é **2.000 por linguagem (8.000 total)** — treino em nuvem.

Inventário atual (aprovados × filas × EN):

```bash
python scripts/inventory_datasets.py
```

Config de volumes: `configs/curadoria.yaml` (`piloto`, `meta_v1` ou `escala_2k`).

## Onde treinar

| Volume pt-BR aprovado | Onde |
|---:|---|
| ≤ 50 (smoke) | M1 — só pipeline |
| 200–800 (piloto) | M1 |
| 800–2.400 | M1 ou nuvem |
| **8.000 (2k/lang)** | **Nuvem QLoRA** |

## Multi-idioma (pt-BR + en)

Cada código curado pode ter **duas** respostas:

| Campo | Idioma |
|---|---|
| `documentation_pt` | pt-BR (seções `### Objetivo`, …) |
| `documentation_en` | en (seções `### Objective`, …) |

```bash
python scripts/build_sft_bilingual.py --config configs/train_bilingual.yaml
# → dataset/processed/sft_bilingual/  (2 linhas SFT por código quando ambos existem)
```

Inferência:

```bash
python scripts/document_code.py --file ... --language php --doc-language pt-BR
python scripts/document_code.py --file ... --language php --doc-language en
```

CodeXGLUE sozinho só gera o lado **en** estruturado; pt-BR exige curadoria (ou `bilingual.mode: hybrid` + docs aprovados).

## Fluxo

```text
CodeXGLUE + Spider
        ↓ normalização e deduplicação
Fila de revisão (status pending)
        ↓ documentação em português + revisão humana
Arquivos curados (status approved)
        ↓ validação de seções e vazamento
Dataset SFT com chat template do Qwen
```

## Regras de curadoria

- A referência pública ajuda a entender a intenção, mas não deve ser copiada como verdade.
- Toda afirmação precisa estar sustentada pelo código apresentado.
- Dúvidas devem ir para `Pontos não determinados`.
- Não inventar validações, efeitos colaterais, tipos ou regras externas.
- Todas as seis seções obrigatórias devem estar presentes.
- `review_status` deve ser alterado para `approved` somente após revisão.
- Treino, validação, teste e benchmark não podem compartilhar o mesmo código.

## Comandos

```bash
python scripts/inventory_datasets.py
python scripts/prepare_sql_spider.py --config configs/curadoria.yaml
python scripts/prepare_dataset.py --config configs/curadoria.yaml
python scripts/create_curated_queue.py --config configs/curadoria.yaml --mode piloto
# após revisão humana → dataset/curated/{train,validation,test}.jsonl
python scripts/build_sft_dataset.py --config configs/curadoria.yaml
```

O último comando falhará se houver exemplos pendentes, seções ausentes ou vazamento para o
benchmark.
