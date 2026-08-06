# Relatório do baseline inicial

## Escopo

O baseline comparou Qwen3-1.7B e Ministral-3-3B em quatro funções, sendo um exemplo de Python,
PHP, JavaScript e SQL por modelo.

| Modelo | Casos | Estrutura | Tempo médio | Tokens médios |
|---|---:|---:|---:|---:|
| **Qwen3-1.7B** | 4 | **87,5%** | **18,650 s** | **292,0** |
| Ministral-3-3B | 4 | 50,0% | 32,297 s | 576,2 |

O Qwen foi aproximadamente 42,3% mais rápido e gerou 49,3% menos tokens. Também apresentou
maior regularidade no formato. O Ministral produziu respostas mais detalhadas em alguns casos,
mas com maior verbosidade e variações nos títulos Markdown.

## Decisão provisória

O Qwen3-1.7B foi selecionado provisoriamente como modelo principal. Como havia somente quatro
exemplos, foi determinada a expansão do teste para 100 exemplos por linguagem.

## Limitações

- apenas um caso por linguagem;
- avaliação humana ainda insuficiente;
- métrica estrutural baseada na presença literal dos títulos;
- pequenas variações de Markdown podiam ser penalizadas;
- o resultado não demonstrava sozinho correção semântica.

O relatório original e detalhado permanece em `docs/RELATORIO_BASELINE.md`.
