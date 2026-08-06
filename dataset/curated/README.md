# Curadoria em português

Esta pasta recebe `train.jsonl`, `validation.jsonl` e `test.jsonl` revisados por humanos.

Cada linha deve conter:

```json
{"id":"exemplo-001","language":"python","code":"def ...","documentation_pt":"## Método ou função\n...","source":"code_x_glue","review_status":"approved"}
```

O campo `documentation_pt` precisa conter todas as seções do contrato documental. Docstrings
originais do CodeXGLUE não devem ser copiadas diretamente para este campo sem tradução,
reestruturação, análise de fidelidade e revisão.

Modelos de contrato e exemplo estão em `templates/`. O checklist de revisão está em
`docs/CHECKLIST_CURADORIA.md`.

