# Relatório de análise do baseline

## 1. Resumo executivo

Foram comparados os modelos `Qwen/Qwen3-1.7B` e
`mistralai/Ministral-3-3B-Instruct-2512-BF16` na geração de documentação técnica
em português para quatro exemplos: Python, PHP, JavaScript e SQL.

**Recomendação provisória:** adotar o **Qwen3 1.7B como modelo-base principal** para o primeiro
ciclo de Fine-Tuning. Ele foi mais rápido, produziu respostas mais curtas e apresentou maior
regularidade no formato solicitado. O Ministral deve permanecer como referência de qualidade,
pois produziu documentação mais detalhada e, em vários casos, semanticamente mais completa.

Essa recomendação ainda não é definitiva. O conjunto atual contém apenas um caso por linguagem
e a planilha de avaliação humana original não foi preenchida.

## 2. Artefatos analisados

- `notebooks/qwen3-1.7b.jsonl`;
- `notebooks/ministral3-3b.jsonl`;
- `notebooks/model_comparison.csv`;
- `notebooks/human_scores.csv`.

Cada modelo recebeu exatamente os mesmos quatro códigos e o mesmo contrato documental.

## 3. Resultados automáticos

| Modelo | Casos | Estrutura exata | Tempo médio | Tokens médios |
|---|---:|---:|---:|---:|
| Qwen3 1.7B | 4 | 87,5% | 18,650 s | 292,0 |
| Ministral 3 3B | 4 | 50,0% | 32,297 s | 576,2 |

Comparado ao Ministral, o Qwen:

- reduziu o tempo médio em aproximadamente **42,3%**;
- gerou aproximadamente **49,3% menos tokens**;
- foi mais consistente com os títulos Markdown exigidos.

### Limitação da métrica estrutural

Os 50% do Ministral não significam que metade da documentação estava ausente. Nos casos Python
e PHP, o modelo escreveu títulos como `### **Objetivo**`. O avaliador automático procurava a
sequência literal `### Objetivo` e registrou `0/6`, embora as seis seções estivessem presentes.

A métrica precisa ser normalizada antes da avaliação final, removendo negrito, numeração e
espaços adicionais dos títulos. No SQL, contudo, houve um problema real: títulos como
`### ### Objetivo` e resposta interrompida ao atingir 700 tokens.

## 4. Análise por linguagem

### Python

**Qwen**

- Explicou corretamente a fórmula e o intervalo permitido para o percentual.
- Inventou a regra de que o valor original deve ser positivo; o código não faz essa validação.
- Classificou os casos de percentual 0 e 100 como não determinados, embora o código defina
  explicitamente ambos como válidos.

**Ministral**

- Documentou corretamente a validação, a exceção e a fórmula.
- Diferenciou melhor a ausência de validação para `valor`.
- Foi mais longo que o necessário, mas apresentou fidelidade superior neste caso.

**Vencedor qualitativo:** Ministral.

### PHP

**Qwen**

- Identificou corretamente as duas condições necessárias para cancelamento.
- Não seguiu integralmente o formato: substituiu Objetivo, Parâmetros e Retorno por uma seção
  chamada `Objeto`.
- Criou pontos não determinados pouco úteis sobre mudanças de estado após a chamada.

**Ministral**

- Explicou claramente que ambas as condições devem ser verdadeiras.
- Cobriu parâmetro, retorno e fluxo de forma mais completa.
- Extrapolou ao interpretar `PENDENTE` como “sem processamento ou pagamento” e ao dizer que o
  pedido nunca foi faturado; o código verifica apenas o estado retornado no momento da chamada.

**Vencedor qualitativo:** Ministral, com ressalvas de inferência.

### JavaScript

**Qwen**

- Descreveu corretamente o filtro por item ativo e estoque maior que zero.
- Foi conciso e seguiu todas as seções.
- Os pontos não determinados contradizem parcialmente a própria análise ao dizer que não há
  evidência para determinar as condições, embora elas estejam explícitas no filtro.

**Ministral**

- Produziu documentação mais completa sobre retorno, condições e casos-limite.
- Separou adequadamente fatos observados de tipos não declarados.
- Acrescentou detalhes além do necessário, aumentando custo e tempo.

**Vencedor qualitativo:** Ministral por pequena margem.

### SQL

**Qwen**

- Explicou corretamente o `LEFT JOIN`, o filtro de pedidos pagos, o agrupamento e o `HAVING`.
- Seguiu todas as seções e permaneceu dentro do limite de geração.
- Os pontos não determinados foram vagos e desnecessários, mas não alteraram a regra principal.

**Ministral**

- Explicou corretamente a intenção geral e o fluxo da consulta.
- Tratou tabelas e condições como parâmetros, o que não é tecnicamente adequado para essa
  consulta sem parâmetros vinculados.
- Duplicou marcadores nos títulos (`### ###`) e atingiu o limite de 700 tokens, deixando a
  documentação incompleta.
- Inferiu que status `PAGO` significa pedido concluído ou validado, algo não comprovado pelo SQL.

**Vencedor qualitativo:** Qwen.

## 5. Auditoria qualitativa inicial

As notas abaixo foram atribuídas após leitura técnica das oito respostas. Elas são uma auditoria
editorial inicial, não uma avaliação humana cega e independente.

| Modelo | Fidelidade | Cobertura | Clareza | Regras | Estrutura |
|---|---:|---:|---:|---:|---:|
| Qwen3 1.7B | 3,75 | 3,75 | 4,00 | 3,75 | 4,50 |
| Ministral 3 3B | 3,50 | 4,75 | 3,75 | 3,50 | 3,50 |

Escala: 1 (insuficiente) a 5 (excelente).

O Ministral teve melhor cobertura, mas perdeu pontos por inferências, verbosidade e problemas de
formato no SQL. O Qwen foi mais equilibrado, embora tenha inventado uma restrição no exemplo
Python. Esse tipo de invenção deve ser tratado como erro crítico no próximo ciclo.

## 6. Conclusão

O **Qwen3 1.7B oferece o melhor equilíbrio atual entre qualidade, velocidade, tamanho da resposta
e aderência ao formato**. Essas características são importantes para Fine-Tuning em Colab e para
processamento posterior de muitas funções de um sistema legado.

O **Ministral 3 3B demonstra maior capacidade de detalhamento**, mas precisa de controle mais
forte de formato, concisão e inferências. Seu custo de inferência também foi maior no teste.

Portanto:

1. usar Qwen3 1.7B como modelo-base do primeiro experimento QLoRA;
2. conservar o Ministral como baseline alternativo;
3. não eliminar o Ministral antes de uma avaliação maior e cega;
4. priorizar no Fine-Tuning a fidelidade e a declaração correta de incertezas.

## 7. Próximas ações

1. Expandir o benchmark para pelo menos 10 exemplos por linguagem.
2. Incluir funções com exceções, efeitos colaterais, dependências e código ambíguo.
3. Corrigir o avaliador estrutural para normalizar títulos Markdown.
4. Reduzir `max_new_tokens` ou reforçar concisão para evitar respostas excessivas.
5. Preencher a rubrica humana por pelo menos dois avaliadores, sem identificar o modelo.
6. Calcular concordância entre avaliadores e registrar erros críticos separadamente.
7. Executar o baseline ampliado antes do treinamento QLoRA.

## 8. Limitações do experimento

- apenas quatro exemplos;
- somente uma execução por exemplo;
- ausência de avaliação humana independente preenchida;
- métrica estrutural baseada em correspondência literal;
- hardware e consumo máximo de memória não registrados no CSV;
- ausência de métricas de fidelidade automatizadas;
- limite de 700 tokens afetou o resultado SQL do Ministral.

