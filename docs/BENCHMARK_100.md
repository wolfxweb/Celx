# Benchmark com 100 exemplos por linguagem

## Composição

| Linguagem | Quantidade | Fonte |
|---|---:|---|
| Python | 100 | CodeXGLUE Code-to-Text |
| PHP | 100 |  Code-to-Text |
| JavaScript | 100 | CodeXGLUE Code-to-Text |
| SQL | 100 | Spider |
| **Total** | **400** | — |

O script `scripts/build_benchmark.py` usa seed 42, remove consultas ou funções duplicadas e
exige exatamente 100 casos únicos para cada linguagem.

## Fontes e licenças

- CodeXGLUE: pares de código e docstring usados para PHP, Python e JavaScript.
- Spider (`xlangai/spider`): perguntas humanas e consultas SQL, licença CC BY-SA 4.0.

A referência original é preservada em `source_reference`. Ela não é tratada como documentação
estruturada em português nem como resposta ideal para métricas textuais diretas.

## Dois níveis de avaliação

### Benchmark completo

Os 400 casos medem:

- estabilidade de geração;
- aderência ao formato;
- respostas incompletas;
- tempo;
- quantidade de tokens;
- comportamento por linguagem.

### Revisão humana amostral

A fidelidade técnica deve ser avaliada em uma amostra estratificada, inicialmente 20 casos por
linguagem e por modelo. Avaliar manualmente 800 respostas completas não é necessário para a
primeira decisão, mas os erros críticos devem ser procurados em todas as linguagens.

## Execução retomável

O script grava uma linha após cada resposta. Se for executado novamente sem `--overwrite`, IDs já
presentes no resultado são ignorados. No Colab, `03_benchmark_100_colab.ipynb` aponta o diretório
de resultados para o Google Drive.

Recomendação operacional:

1. executar o Qwen em uma sessão;
2. executar o Ministral em outra sessão;
3. repetir uma sessão interrompida sem apagar arquivos;
4. gerar o comparativo quando ambos chegarem a 400 respostas.

