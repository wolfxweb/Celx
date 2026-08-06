# Fine-Tuning para documentação de software legado

Projeto experimental para especializar um LLM na geração de documentação técnica em português
a partir de código legado em PHP, Python, JavaScript e SQL.

O foco é explicar **como o código existente funciona** e registrar regras de negócio observáveis.
O modelo deve declarar incertezas em vez de inventar requisitos.

## Estado do projeto

O projeto está no **Marco 1 — Fundação**. A estrutura, o formato documental e os primeiros
pipelines foram criados. O dataset ainda não foi baixado e nenhum treinamento foi executado.

Consulte o [cronograma](docs/CRONOGRAMA.md), o [escopo](docs/ESCOPO.md) e o
[protocolo de seleção do modelo](docs/SELECAO_MODELO.md).

## Saída esperada

```markdown
## Método ou função

Calcula o valor total de um pedido.

### Objetivo

Somar os valores dos itens pertencentes ao pedido.

### Parâmetros

- `pedido`: objeto que contém os itens processados.

### Retorno

Valor total no formato `BigDecimal`.

### Funcionamento

1. Inicializa o total com zero.
2. Percorre os itens do pedido.
3. Soma o valor de cada item.
4. Retorna o total.

### Regras de negócio identificadas

- O total corresponde à soma dos valores dos itens.

### Pontos não determinados

- O comportamento para pedido ou valores nulos não está definido.
```

## Estrutura

```text
.
├── configs/       # parâmetros de dados, modelo e treinamento
├── dataset/       # dados locais (não versionados)
├── docs/          # escopo, decisões e cronograma
├── evaluation/    # avaliação automática e humana
├── inference/     # uso do modelo-base ou adapter
├── legacy_doc/    # componentes Python reutilizáveis
├── models/        # adapters e modelos locais (não versionados)
├── notebooks/     # notebooks reproduzíveis no Colab
├── presentation/  # apresentação final
├── scripts/       # aquisição e preparação dos dados
├── tests/         # testes rápidos sem GPU
└── training/      # treinamento LoRA/QLoRA
```

## Requisitos

- Python 3.10 ou superior;
- GPU NVIDIA/CUDA para QLoRA;
- Google Colab é o ambiente previsto para treinamento;
- conta Hugging Face apenas para publicação ou modelos restritos.

## Instalação local

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

No Windows PowerShell, ative o ambiente com:

```powershell
.venv\Scripts\Activate.ps1
```

## Fluxo de execução

Para executar a seleção do modelo no Google Colab, abra
`notebooks/01_baseline_colab.ipynb`, ative uma GPU T4 e siga as células. Enquanto o projeto não
estiver publicado em um repositório remoto, o notebook recebe um arquivo ZIP da pasta local.

### 1. Validar a fundação

```bash
python -m pytest
python -m ruff check .
```

### 2. Preparar o CodeXGLUE

Este comando baixa dados e o tokenizer do Hugging Face:

```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```

Os dados preparados serão gravados em `dataset/processed/` junto com um manifesto.

### 3. Treinar o adapter QLoRA

Execute em uma sessão com GPU CUDA:

```bash
python training/train_qlora.py --config configs/default.yaml
```

### 4. Documentar uma função

Com o modelo-base:

```bash
python inference/document_code.py \
  --file dataset/examples/calcula_total.py \
  --language python
```

Com um adapter treinado:

```bash
python inference/document_code.py \
  --file dataset/examples/calcula_total.py \
  --language python \
  --adapter models/qwen3-legacy-doc-qlora
```

## Modelo e dataset iniciais

- Modelo configurado: `Qwen/Qwen3-1.7B`.
- Dataset: `google/code_x_glue_ct_code_to_text`.
- Linguagens CodeXGLUE: PHP, Python e JavaScript.
- Linguagem complementar: SQL, com pipeline de dados próprio.

Esses padrões ficam em `configs/default.yaml` e podem ser substituídos pela linha de comando.
A escolha final do modelo será registrada após o baseline comparativo da Semana 2.

## Limitação importante do dataset

O CodeXGLUE oferece pares de código e docstring, mas docstrings não contêm necessariamente
regras de negócio completas. A primeira etapa ensina o modelo a relacionar código e descrição.
SQL não faz parte desse dataset e exigirá uma fonte pública complementar ou exemplos próprios
devidamente licenciados. Depois será criado um subconjunto curado, em português, para ensinar o formato documental e a
separação entre evidência e incerteza.

## Princípios de qualidade

- Não apresentar inferências como regras confirmadas.
- Não assumir comportamento para nulos, exceções ou integrações ausentes.
- Manter rastreabilidade entre documentação e código analisado.
- Avaliar fidelidade, clareza, cobertura e consistência do formato.
- Comparar o adapter ao mesmo modelo-base sem Fine-Tuning.
