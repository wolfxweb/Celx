# Celx — documentação de software legado

Especializar um LLM para analisar código-fonte legado e gerar **documentação técnica** (pt-BR e/ou en) que ajude pessoas desenvolvedoras a entender o funcionamento do sistema e as regras de negócio **sustentadas pelo código**.

O modelo explica o comportamento observável e declara incertezas. **Não inventa** requisitos nem regras que o trecho não sustente.

## Objetivo

| | |
|---|---|
| **Entrada** | Trecho de código (função/método PHP ou JavaScript, ou consulta SQL) |
| **Saída** | Markdown estruturado: objetivo, parâmetros, retorno, funcionamento, regras identificáveis, pontos não determinados |
| **Modelo** | `Qwen/Qwen3-1.7B` + adapter LoRA/QLoRA Celx |
| **Critério** | Fiel ao código; incompleto + incerteza explícita é melhor que alucinação |

## Estado atual

- Treino QLoRA (~300k PHP + JavaScript + SQL) concluído no RunPod.
- Pacote local: `models/export/qwen3-1.7b-celx/adapter/` (a partir de `models/celx-adapter-local.tgz`).
- Snapshot do ciclo anterior em [`arquivos/`](arquivos/) (histórico; não editar).

## Formato da documentação

1. Método ou função / Consulta SQL  
2. Objetivo  
3. Parâmetros  
4. Retorno  
5. Funcionamento  
6. Regras de negócio identificadas  
7. Pontos não determinados  

## Bases de dados

| Fonte | Uso | URL |
|---|---|---|
| **CodeXGLUE** Code-to-Text (PHP, JavaScript; Python opcional) | Pares código → texto para SFT | https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text |
| **Spider** | Exemplos SQL | https://huggingface.co/datasets/xlangai/spider |
| **Modelo base** | Checkpoint a adaptar | https://huggingface.co/Qwen/Qwen3-1.7B |
| Curadoria Celx | Exemplos revisados (pt-BR / bilingue) | `dataset/curated/` |

Volume típico no treino cheio (sem Python): ~241k PHP + ~58k JS + ~5–7k SQL ≈ **~300k** exemplos.

## Resultado esperado

- Adapter em `models/export/qwen3-1.7b-celx/adapter/`
- Inferência local via `scripts/document_code.py` (Mac: LoRA/MPS; sem QLoRA 4-bit)
- Eval estrutural das seções no split de teste
- Opcional: API OpenAI-compatible (`scripts/start_celx_api.sh`) e Continue no VS Code

## Estrutura

```text
.
├── arquivos/      # histórico congelado (não editar)
├── configs/       # treino, curadoria, candidatos de modelo
├── dataset/       # exemplos, SQL, curadoria, SFT processado
├── legacy_doc/    # prompts e config Python
├── models/        # adapters / export (gitignored)
├── notebooks/     # 01 benchmark → 04 pipeline completo
├── outputs/       # métricas e eval (gitignored)
├── scripts/       # CLI: dados, treino, eval, inferência, export
└── tests/
```

## Requisitos

- Python 3.10+ (venv do projeto)
- **Mac (MPS):** inferência e treinos pequenos; não use bitsandbytes/QLoRA 4-bit
- **GPU NVIDIA (RunPod/Colab):** treino cheio (~300k) com QLoRA

## Instalação local

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usar o adapter no Mac

Se ainda não extraiu o pacote baixado do RunPod:

```bash
mkdir -p models/export
tar -xzf models/celx-adapter-local.tgz -C models/export
# → models/export/qwen3-1.7b-celx/adapter/
```

Inferência (na primeira vez baixa a base `Qwen/Qwen3-1.7B` do Hugging Face):

```bash
source .venv/bin/activate
python scripts/document_code.py \
  --file dataset/examples/calcula_total.php \
  --language php \
  --doc-language pt-BR \
  --adapter models/export/qwen3-1.7b-celx/adapter \
  --config configs/train_full.yaml
```

Inglês: `--doc-language en`.

### Editor / API

| Ferramenta | Como |
|---|---|
| **Terminal** | `document_code.py` com o adapter acima |
| **VS Code + Continue** | API local ou Ollama (ver `.continue/`) |
| **Cursor** | `localhost` costuma ser bloqueado; use Terminal/VS Code para o adapter local |

```bash
# API OpenAI-compatible (porta 8000) — Continue / clientes HTTP
bash scripts/start_celx_api.sh
```

## Treinar de novo (RunPod)

1. Envie só `notebooks/04_pipeline_completo.ipynb` (cria `/workspace/celx`).
2. Stages: **A** dados+SFT+treino+eval → **B** (opcional bilingue) → **Final** export → **C** smoke.
3. Baixe `models/export/celx-adapter-local.tgz` (~29 MB) de volta ao Mac.

Config CUDA: `configs/train_php_js_sql_full.yaml`.

Alternativa local/scripts:

```bash
source .venv/bin/activate
bash scripts/pipeline_full.sh configs/train_full.yaml          # amostra (Mac)
# ou na GPU:
bash scripts/pipeline_full.sh configs/train_php_js_sql_full.yaml
python scripts/watch_training.py
```

## Notebooks

| Notebook | Função |
|---|---|
| `01_benchmark.ipynb` | Seleção / baseline do modelo |
| `02_treino_qlora.ipynb` | Smoke do pipeline |
| `03_treino_real.ipynb` | Treino CodeXGLUE → LoRA |
| `04_pipeline_completo.ipynb` | **End-to-end** (recomendado na nuvem) |
