# Dados SQL

O CodeXGLUE Code-to-Text não possui SQL. Este diretório recebe exemplos gerados por:

```bash
python scripts/prepare_sql_spider.py --config configs/train_full.yaml
```

Isso baixa `xlangai/spider` e grava `train.jsonl`, `validation.jsonl`, `test.jsonl`.

## Formato

```json
{"id":"...","language":"sql","code":"SELECT ...","reference":"...","source":"xlangai/spider","license":"CC-BY-SA-4.0"}
```

O notebook `04` / `prepare_dataset.py` mescla esses arquivos com PHP/Python/JavaScript do CodeXGLUE.
