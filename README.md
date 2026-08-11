# Celx — documentação de software legado

Fine-tuning experimental de um LLM para gerar documentação técnica em português a partir de
código legado (PHP, Python, JavaScript e SQL).

O modelo deve explicar o comportamento observável e declarar incertezas — sem inventar regras.

## Estado

Projeto **reiniciado**. Snapshot anterior em [`arquivos/`](arquivos/).

Ciclo atual: **Fase 2 — Benchmark** para escolher o modelo-base.
Plano completo: [`docs/PLANO.md`](docs/PLANO.md).

## Estrutura

```text
.
├── arquivos/      # histórico congelado (não editar)
├── configs/       # modelo, dados e treinamento
├── dataset/       # exemplos, benchmark e curadoria
├── docs/          # plano, escopo, rubrica, decisões
├── legacy_doc/    # prompts e config Python
├── models/        # adapters (gitignored)
├── notebooks/     # 01–04 (benchmark → pipeline completo)
├── outputs/       # resultados de execução (gitignored)
├── scripts/       # CLI: dados, baseline, treino, inferência
└── tests/
```

## Requisitos

- Python 3.10+
- GPU NVIDIA/CUDA para benchmark completo e QLoRA (Kaggle, Colab ou RunPod)
- Este Mac prepara dados e empacota o projeto; não treina

## Instalação local

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Próximo passo

1. Kernel `.venv` → `notebooks/01_benchmark.ipynb` (opcional)
2. **Pipeline completo** → `notebooks/04_pipeline_completo.ipynb`  
   (CodeXGLUE + SQL → treino → eval → export)
3. Alternativas: `03_treino_real.ipynb` (só CodeXGLUE) ou `02` (smoke)

### Usar o modelo no editor

| Editor | Como |
|---|---|
| **VS Code** | Extensão **Continue** + Ollama local (`qwen2.5:1.5b-celx`) e/ou API Celx |
| **Cursor** | `localhost` é bloqueado; use modelos cloud do Cursor, ou VS Code para local |
| **Terminal** | `document_code.py` com o adapter |

```bash
# App Ollama deve estar aberto
# API do adapter Celx (para Continue no VS Code):
bash scripts/start_celx_api.sh
```

Config Continue do projeto: `.continue/config.json`  
- `Ollama Qwen 1.5B Celx` → `qwen2.5:1.5b-celx` @ `http://127.0.0.1:11434`  
- `Celx Legacy Doc` → `http://127.0.0.1:8000/v1` (com a API acima rodando)

```bash
source .venv/bin/activate
bash scripts/pipeline_full.sh
# ou acompanhar treino:
python scripts/watch_training.py
```

Ver `docs/PLANO.md`.

## Modelo e dados previstos

- Candidatos: `Qwen/Qwen3-1.7B` e Ministral 3B (`configs/model_candidates.yaml`)
- Dataset: CodeXGLUE code-to-text + SQL complementar + curadoria em português
