# Relatório comparativo — benchmark ampliado

## Resumo executivo

O teste ampliado comparou os modelos em 400 exemplos por modelo, cobrindo JavaScript, PHP,
Python e SQL.

| Modelo | Exemplos | Estrutura | Tempo médio | Tokens médios |
|---|---:|---:|---:|---:|
| **Qwen3-1.7B** | 400 | **96,08%** | **26,397 s** | **424,6** |
| Ministral-3-3B | 400 | 64,42% | 39,043 s | 680,3 |

Comparado ao Ministral, o Qwen:

- obteve 31,66 pontos percentuais a mais de aderência estrutural;
- foi 32,39% mais rápido;
- gerou 37,59% menos tokens;
- economizou aproximadamente 1 h 24 min nas 400 gerações sequenciais;
- produziu aproximadamente 102.280 tokens a menos no conjunto.

## Conclusão

O Qwen3-1.7B foi confirmado como modelo-base para o treinamento QLoRA. A escolha é sustentada
por regularidade estrutural, velocidade, tamanho das respostas e menor exigência computacional.

## Limitações e próximo controle

O arquivo consolidado não apresenta métricas individuais por linguagem nem mede correção
semântica. Deve-se avaliar uma amostra estratificada usando fidelidade ao código, cobertura das
regras, ausência de invenções, clareza e formato. O benchmark deve permanecer separado do
dataset SFT para evitar contaminação.

O relatório metodológico completo permanece em
`docs/RELATORIO_COMPARATIVO_BENCHMARK_400.md`.
