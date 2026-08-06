# Relatório comparativo — benchmark com 400 exemplos

## Resumo

Foram comparados `Qwen/Qwen3-1.7B` e
`mistralai/Ministral-3-3B-Instruct-2512-BF16` em 400 exemplos por modelo, cobrindo
JavaScript, PHP, Python e SQL.

| Modelo | Exemplos | Estrutura | Tempo médio | Tokens médios |
|---|---:|---:|---:|---:|
| **Qwen3-1.7B** | 400 | **96,08%** | **26,397 s** | **424,6** |
| Ministral-3-3B | 400 | 64,42% | 39,043 s | 680,3 |

Comparado ao Ministral, o Qwen apresentou:

- 31,66 pontos percentuais a mais de aderência à estrutura;
- redução de 32,39% no tempo médio;
- redução de 37,59% nos tokens gerados;
- economia estimada de 1 h 24 min nas 400 gerações sequenciais.

## Decisão

O **Qwen3-1.7B foi selecionado como modelo-base para o Fine-Tuning com QLoRA**. Ele é
mais regular no formato, mais rápido, mais compacto e mais econômico para treinamento e
inferência.

## Limitação

A métrica de estrutura verifica a presença literal das seções esperadas, mas não comprova que
as regras de negócio descritas estão corretas. O CSV também não contém resultados separados por
linguagem. Antes da conclusão final, deve ser feita avaliação humana estratificada, verificando:

- fidelidade ao código;
- cobertura do comportamento observável;
- ausência de regras inventadas;
- clareza da documentação;
- aderência ao formato.

O relatório metodológico completo está em
`docs/RELATORIO_COMPARATIVO_BENCHMARK_400.md`.
