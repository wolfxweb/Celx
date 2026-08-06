# Pacote reproduzível — comparação de modelos

Esta subpasta concentra os arquivos necessários para reproduzir e analisar o teste de modelos
do projeto de documentação automática de código legado.

## Estrutura

```text
pacote_teste_modelos/
├── 01_executar_comparativo.ipynb
├── README.md
├── requirements_teste.txt
├── avaliacao/
│   ├── RUBRICA_AVALIACAO.md
│   └── human_scores.csv
├── relatorios/
│   ├── RELATORIO_BASELINE.md
│   └── RELATORIO_COMPARATIVO_BENCHMARK_400.md
└── resultados/
    ├── model_comparison_baseline.csv
    └── model_comparison_400.csv
```

## O que está incluído

- notebook único para validar o ambiente, executar os modelos e gerar o comparativo;
- dependências mínimas do teste;
- resultados consolidados do baseline e do benchmark ampliado;
- ficha e rubrica para avaliação humana;
- relatórios das duas fases já realizadas.

## Pré-requisitos

- projeto Celx completo, pois o notebook utiliza `scripts/run_baseline.py`,
  `evaluation/compare_models.py`, `configs/model_candidates.yaml` e o pacote `legacy_doc`;
- GPU NVIDIA com CUDA;
- acesso à internet para baixar os modelos públicos do Hugging Face;
- Python 3.10 ou 3.11.

Os modelos usados não exigem token para download público. Um `HF_TOKEN` é opcional e apenas
ajuda a evitar limites de requisições.

## Execução no RunPod

1. Use o template oficial RunPod PyTorch 2.8.0.
2. Coloque o projeto em `/workspace/legacy-doc-project`.
3. Abra `01_executar_comparativo.ipynb` no JupyterLab.
4. Execute as células em ordem.
5. Selecione apenas um modelo por vez para controlar VRAM e custo.
6. Depois que os dois JSONL existirem, execute a célula de comparação.
7. Baixe ou faça backup da pasta de resultados antes de encerrar o Pod.

## Critério de decisão

O comparativo automático mede aderência estrutural, tempo médio e tokens. A decisão final deve
também usar a rubrica humana, porque presença de títulos não comprova fidelidade ao código nem
correção das regras de negócio.

## Resultado atual

No benchmark de 400 exemplos, o Qwen3-1.7B atingiu 96,08% de aderência estrutural, contra
64,42% do Ministral. Também foi 32,39% mais rápido e gerou 37,59% menos tokens. Por isso, foi
selecionado para a fase de Fine-Tuning QLoRA.

## Próxima etapa: dataset SFT

Execute `notebooks/etapa_05_dataset_sft/05_preparacao_dataset_sft.ipynb`. Ele cria uma fila piloto de 280 exemplos
balanceados, bloqueia códigos presentes no benchmark e exige aprovação humana antes de produzir
o dataset utilizado pelo treinamento.
