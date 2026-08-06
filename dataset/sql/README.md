# Dados SQL

O CodeXGLUE Code-to-Text não possui uma configuração SQL. Por isso, os dados SQL ficam
separados para preservar a procedência e a licença de cada exemplo.

## Formato esperado

Cada divisão deverá ser um arquivo JSON Lines: `train.jsonl`, `validation.jsonl` e `test.jsonl`.

```json
{"id":"sql-001","language":"sql","code":"SELECT ...","reference":"Lista ...","source":"...","license":"..."}
```

Campos obrigatórios:

- `id`: identificador estável;
- `language`: sempre `sql`;
- `code`: consulta, view, function, trigger ou procedure;
- `reference`: descrição fiel do comportamento;
- `source`: origem rastreável;
- `license`: licença que permita o uso no treinamento.

Nenhum dado SQL será incluído no treinamento antes da validação de origem, licença, duplicatas
e qualidade da descrição. Os exemplos em `dataset/examples/` são demonstrações do projeto, não
constituem um dataset de treinamento.

