# Etapa 05 — preparação do dataset SFT

Esta pasta reúne os materiais da fase que prepara os exemplos supervisionados usados no
Fine-Tuning QLoRA do Qwen3-1.7B.

## Conteúdo

```text
etapa_05_dataset_sft/
├── 05_preparacao_dataset_sft.ipynb
├── README.md
├── CHECKLIST_CURADORIA.md
├── ARQUIVOS_UTILIZADOS.md
└── templates/
    ├── exemplo_curado.jsonl
    └── contrato_documentacao.md
```

## Objetivo

Criar um dataset em português com PHP, Python, JavaScript e SQL que ensine o modelo a explicar
objetivo, parâmetros, retorno, funcionamento, regras sustentadas pelo código e pontos que não
podem ser determinados somente pelo trecho analisado.

## Piloto

| Divisão | Python | PHP | JavaScript | SQL | Total |
|---|---:|---:|---:|---:|---:|
| Treino | 50 | 50 | 50 | 50 | 200 |
| Validação | 10 | 10 | 10 | 10 | 40 |
| Teste | 10 | 10 | 10 | 10 | 40 |

Depois do piloto, a meta recomendada é 2.400 exemplos: 500 de treino, 50 de validação e 50 de
teste por linguagem.

## Ordem de execução

1. Abra `05_preparacao_dataset_sft.ipynb` no RunPod ou Colab.
2. Normalize os datasets públicos.
3. Crie ou reutilize o benchmark reservado de 400 casos.
4. Gere as filas em `dataset/curated/queues/`.
5. Preencha a documentação em português usando o contrato desta pasta.
6. Faça a revisão humana e altere `review_status` para `approved`.
7. Salve os aprovados como `train.jsonl`, `validation.jsonl` e `test.jsonl`.
8. Construa o dataset SFT e confira seu manifesto.

## Proteções

- exclusão por hash dos exemplos presentes no benchmark;
- deduplicação entre treino, validação e teste;
- exigência de todas as seções;
- bloqueio de registros sem aprovação humana;
- separação entre a referência pública e a documentação final.

## Saída esperada

```text
dataset/processed/sft/
├── train/
├── validation/
├── test/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── manifest.json
```

Depois dessa validação, a próxima fase será o treinamento no RunPod.
