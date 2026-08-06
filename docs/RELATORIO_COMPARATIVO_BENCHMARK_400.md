# Relatório comparativo dos modelos — benchmark com 400 exemplos

## 1. Resumo executivo

O benchmark comparou `Qwen/Qwen3-1.7B` e
`mistralai/Ministral-3-3B-Instruct-2512-BF16` em **400 exemplos por modelo**, cobrindo
JavaScript, PHP, Python e SQL.

**Recomendação:** manter o **Qwen3-1.7B como modelo-base do Fine-Tuning**. Ele apresentou
o melhor resultado em todas as métricas automáticas disponíveis: maior aderência à estrutura
documental, menor tempo de geração e menor quantidade de tokens.

O resultado confirma, em uma amostra muito maior, a escolha provisória feita no baseline de
quatro exemplos. Entretanto, a decisão mede principalmente eficiência e padronização. Antes de
afirmar que o Qwen compreende melhor as regras de negócio, ainda é necessária uma avaliação
semântica, preferencialmente humana, sobre uma amostra estratificada.

## 2. Fonte e escopo

Arquivo analisado: `notebooks/model_comparison (2).csv`.

| Item | Valor |
|---|---|
| Modelos | Qwen3-1.7B e Ministral-3-3B-Instruct |
| Exemplos | 400 por modelo |
| Linguagens | JavaScript, PHP, Python e SQL |
| Métricas | estrutura, tempo médio e tokens gerados |

O CSV consolidado não contém resultados separados por linguagem nem notas humanas. Portanto,
este relatório não atribui vencedores individuais para PHP, Python, JavaScript ou SQL.

## 3. Resultados

| Modelo | Exemplos | Estrutura | Tempo médio | Tokens médios |
|---|---:|---:|---:|---:|
| **Qwen3-1.7B** | 400 | **96,08%** | **26,397 s** | **424,6** |
| Ministral-3-3B | 400 | 64,42% | 39,043 s | 680,3 |

### Diferenças observadas

Comparado ao Ministral, o Qwen:

- obteve **31,66 pontos percentuais** a mais de aderência estrutural;
- foi **32,39% mais rápido**, economizando em média 12,646 segundos por exemplo;
- gerou **37,59% menos tokens**, uma redução média de 255,7 tokens por resposta.

Considerando as 400 gerações sequenciais registradas, os valores médios correspondem
aproximadamente a:

| Modelo | Tempo acumulado estimado | Tokens totais estimados |
|---|---:|---:|
| Qwen3-1.7B | 2 h 56 min | 169.840 |
| Ministral-3-3B | 4 h 20 min | 272.120 |
| **Economia com Qwen** | **1 h 24 min** | **102.280** |

Esses totais são projeções calculadas a partir das médias do CSV, não medições adicionais.

## 4. Interpretação

### Aderência à estrutura

A pontuação estrutural verifica a presença literal das seis seções esperadas:

1. Objetivo;
2. Parâmetros;
3. Retorno;
4. Funcionamento;
5. Regras de negócio identificadas;
6. Pontos não determinados.

Os **96,08%** do Qwen indicam uma saída muito mais previsível e adequada para automação,
indexação e uso posterior como referência de especificação. Os **64,42%** do Ministral sugerem
maior variação no formato.

Essa métrica não avalia se o conteúdo de cada seção está correto. Além disso, pequenas variações
de Markdown, como negrito ou numeração no título, podem ser penalizadas mesmo quando a seção
existe. Por isso, ela deve ser interpretada como aderência ao contrato de saída, não como nota
geral de qualidade.

### Desempenho

O Qwen apresentou menor latência apesar de produzir documentação suficientemente estruturada.
Isso favorece o processamento em lote de sistemas legados e reduz o custo de GPU por função.
A comparação de tempo, contudo, só é rigorosamente válida se os dois modelos tiverem sido
executados no mesmo hardware e sob configurações equivalentes.

### Tamanho das respostas

O Qwen produziu respostas mais compactas. Isso é positivo para o objetivo do projeto, desde que
a redução não elimine regras importantes. O Ministral gerou cerca de 60% mais tokens que o Qwen,
mas o CSV não permite determinar se esse conteúdo adicional representa maior cobertura semântica
ou apenas maior verbosidade.

## 5. Decisão para a próxima fase

O **Qwen3-1.7B está aprovado como modelo-base para o treinamento QLoRA** porque:

- mantém o formato documental com alta regularidade;
- tem menor latência;
- usa menos tokens;
- é menor e mais econômico para treinamento e inferência;
- apresentou vantagem consistente no baseline inicial e no benchmark ampliado.

O Ministral pode permanecer apenas como referência externa de comparação, sem necessidade de
participar do treinamento principal.

## 6. Próximas atividades recomendadas

1. Avaliar manualmente uma amostra estratificada de 40 exemplos: 10 por linguagem.
2. Pontuar fidelidade ao código, cobertura das regras, ausência de invenções, clareza e formato.
3. Preparar o dataset SFT em português com exemplos revisados.
4. Executar QLoRA no Qwen3-1.7B, mantendo um conjunto de validação não usado no treinamento.
5. Repetir o benchmark antes e depois do Fine-Tuning com os mesmos parâmetros de geração.
6. Comparar o modelo-base e o modelo ajustado tanto por métricas automáticas quanto por revisão
   humana.

## 7. Conclusão

O benchmark ampliado oferece evidência suficiente para escolher o **Qwen3-1.7B como a opção mais
eficiente e consistente estruturalmente**. A próxima etapa deve verificar se essa vantagem também
se mantém na fidelidade das regras de negócio. O Fine-Tuning não deve começar apenas para melhorar
o estilo; ele deve ensinar o modelo a documentar o comportamento observável sem inventar regras
que não estejam sustentadas pelo código.
