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

Para um ensaio barato, comece com 50/10/10 por linguagem. A meta de 2.400 exemplos é a primeira
versão recomendada, não uma exigência técnica.

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
python -m scripts.prepare_dataset
python -m scripts.build_benchmark
python -m scripts.create_curated_queue
python -m scripts.build_sft_dataset
```

O último comando falhará se houver exemplos pendentes, seções ausentes ou vazamento para o
benchmark.
