# Fase de dados

## Objetivo

Construir um dataset supervisionado confiável para o Qwen3 1.7B sem confundir docstrings em
inglês com documentação técnica estruturada em português.

## Três camadas

1. **Normalizada:** código e referência original do CodeXGLUE, limpos e deduplicados.
2. **Curada:** documentação em português revisada e fiel ao código.
3. **SFT:** conversas renderizadas com o chat template do Qwen e prontas para treinamento.

```text
CodeXGLUE → dataset/processed/codexglue
                    ↓ curadoria
           dataset/curated/*.jsonl
                    ↓ validação e template
              dataset/processed/sft
```

SQL segue uma fonte e curadoria separadas, mas entra no mesmo esquema curado antes do SFT.

Para avaliação, o Spider fornece 100 consultas SQL acompanhadas de perguntas humanas. Assim, o
benchmark expandido totaliza 400 casos, sendo 100 por linguagem.

## Comandos

```bash
python scripts/prepare_dataset.py
python scripts/analyze_dataset.py
python scripts/build_benchmark.py
python scripts/build_sft_dataset.py
```

O terceiro comando somente deve ser executado quando as três divisões curadas existirem.

## Critérios de aceite

- nenhuma duplicata por hash entre treino, validação e teste;
- origem e licença rastreáveis;
- todas as respostas em português;
- todas as seções obrigatórias presentes;
- regras de negócio sustentadas pelo código;
- exemplos SQL com fonte separada e licença registrada;
- amostra revisada por pelo menos duas pessoas antes do treinamento final.
