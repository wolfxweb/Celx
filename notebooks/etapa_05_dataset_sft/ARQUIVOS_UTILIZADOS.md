# Arquivos utilizados

Os executáveis permanecem nas pastas oficiais para evitar cópias divergentes.

| Arquivo | Função |
|---|---|
| `configs/default.yaml` | Modelo, diretórios e limites. |
| `scripts/prepare_dataset.py` | Normaliza e separa o CodeXGLUE. |
| `scripts/build_benchmark.py` | Reserva os 400 casos de avaliação. |
| `scripts/create_curated_queue.py` | Cria as filas e exclui o benchmark. |
| `scripts/build_sft_dataset.py` | Valida aprovados e aplica o template do Qwen. |
| `legacy_doc/prompts.py` | Prompt e contrato documental. |
| `docs/CURADORIA_SFT.md` | Metodologia completa. |

## Comandos

```bash
python -m scripts.prepare_dataset
python -m scripts.build_benchmark
python -m scripts.create_curated_queue --train-per-language 50 --validation-per-language 10 --test-per-language 10
python -m scripts.build_sft_dataset
```

O último comando somente deve rodar depois da aprovação humana.
