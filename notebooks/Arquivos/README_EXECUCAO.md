# Guia dos notebooks e resultados

Esta pasta reúne os notebooks reproduzíveis, os resultados executados e os relatórios do
projeto de documentação automática de software legado.

## Arquivos principais

| Arquivo | Finalidade |
|---|---|
| `01_baseline_colab.ipynb` | Executa o baseline inicial dos modelos. |
| `02_dados_codexglue_colab.ipynb` | Obtém e prepara os dados públicos. |
| `03_benchmark_100_colab.ipynb` | Executa 100 exemplos por linguagem no Colab. |
| `03_benchmark_100_kaggle.ipynb` | Versão do benchmark adaptada ao Kaggle. |
| `04_treinamento_qwen3_runpod.ipynb` | Valida o ambiente e executa QLoRA no RunPod. |
| `model_comparison_400.csv` | Comparativo consolidado e renomeado. |
| `qwen3-1.7b.jsonl` | Respostas do baseline inicial do Qwen. |
| `ministral3-3b.jsonl` | Respostas do baseline inicial do Ministral. |
| `human_scores.csv` | Planilha para avaliação humana. |
| `RELATORIO_BASELINE.md` | Análise do baseline inicial. |
| `RELATORIO_COMPARATIVO_BENCHMARK_400.md` | Síntese do benchmark ampliado. |

O arquivo `model_comparison (2).csv` foi preservado como resultado original. A cópia
`model_comparison_400.csv` possui nome estável para uso em scripts e documentação.

## Ordem de execução do projeto

1. Execute o notebook 02 para preparar os dados públicos.
2. Execute o notebook 01 para validar modelos e ambiente com poucos exemplos.
3. Execute uma das versões do notebook 03 para o benchmark ampliado.
4. Analise `model_comparison_400.csv` e preencha a avaliação humana.
5. Revise os exemplos em português e construa o dataset SFT.
6. Execute o notebook 04 no RunPod para treinar o Qwen3-1.7B.
7. Repita o benchmark com o adapter treinado e compare antes/depois.

## Como executar no RunPod

### 1. Criar o Pod

Use o template oficial:

```text
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

Configuração sugerida:

- uma RTX A5000 de 24 GB;
- Container Disk de 40 a 50 GB;
- Volume Disk de 60 GB montado em `/workspace`;
- JupyterLab habilitado.

### 2. Enviar o projeto

Abra `Connect > JupyterLab` e envie o ZIP do projeto para `/workspace`. Os arquivos que
precisam sobreviver ao `Stop` devem permanecer nesse volume.

### 3. Abrir o notebook

Abra `notebooks/04_treinamento_qwen3_runpod.ipynb` e execute as células em ordem. O notebook:

- verifica CUDA, GPU e VRAM;
- localiza ou extrai o projeto;
- instala as bibliotecas necessárias;
- valida o dataset SFT;
- inicia o treinamento;
- confirma a criação do adapter e das métricas.

### 4. Dataset exigido

Antes do treinamento devem existir:

```text
dataset/processed/sft/train/
dataset/processed/sft/validation/
```

Se não existirem, prepare primeiro exemplos curados no formato esperado pelo script
`scripts/build_sft_dataset.py`. Não use o conjunto do benchmark como treinamento, pois isso
contaminaria a avaliação.

### 5. Resultados persistentes

O treinamento grava o adapter em:

```text
models/qwen3-legacy-doc-qlora/
```

Antes de parar o Pod, confirme a existência de `adapter_config.json`, dos pesos do adapter,
do tokenizer e de `metrics.json`. Faça também um backup externo.

## Como interpretar o comparativo

`structure_percent` mede a presença dos títulos solicitados. `average_seconds` mede a latência
média, e `average_generated_tokens` mede o tamanho médio da resposta. Nenhuma dessas métricas,
isoladamente, garante que as regras de negócio estejam semanticamente corretas.
