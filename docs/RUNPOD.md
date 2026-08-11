# RunPod + Jupyter (treino Celx)

Fluxo recomendado para a **A4500 20 GB** (ou outra GPU CUDA) com Jupyter Notebook.

## 1) Criar o pod

1. [RunPod](https://www.runpod.io/) → **Pods** → **Deploy**
2. GPU: **RTX A4500** (20 GB) ou similar
3. Template: **RunPod PyTorch** (ou “Jupyter”) com Jupyter ligado
4. Volume / disco: **≥ 40 GB** (melhor 50–80 GB se for baixar ~300k exemplos)
5. Cole sua SSH key pública (`~/.ssh/id_ed25519_runpod.pub`) em **Settings → SSH Public Keys**
6. Deploy → espere ficar **Running**

## 2) Empacotar o projeto (no Mac)

```bash
cd /Users/wolfx/Documents/Dev/Celx
bash scripts/package_project.sh
# → dist/Celx-colab.zip
```

Envie o zip para o pod (uma das opções):

**Jupyter file browser:** Upload → `/workspace/Celx-colab.zip`

**scp** (troque IP/porta pelo Connect do RunPod):

```bash
scp -P <PORTA> -i ~/.ssh/id_ed25519_runpod \
  dist/Celx-colab.zip root@<IP>:/workspace/Celx-colab.zip
```

**git** (se o repo estiver no GitHub):

```bash
cd /workspace && git clone <seu-repo> Celx
```

## 3) Abrir o Jupyter

No painel do pod: **Connect → HTTP services → Jupyter** (ou a URL que o template mostrar).

1. Abra `notebooks/04_pipeline_completo.ipynb`
2. Rode a célula **0) Setup** — ela acha `/workspace/Celx` ou descompacta o zip
3. Rode as deps e o **Stage A** em ordem

Com CUDA o notebook escolhe sozinho:

- Config: `configs/train_php_js_sql_full.yaml`
- Treino: QLoRA (`scripts/train_qlora.py`)
- Linguagens: **PHP + JavaScript + SQL** (sem Python)

## 4) Acompanhar o treino

No Jupyter: **New → Terminal**

```bash
cd /workspace/Celx   # ou o ROOT impresso no setup
python scripts/watch_training.py
# ou:
tail -f outputs/training_php_js_sql/STATUS.txt
```

## 5) Baixar o resultado

Artefatos importantes:

| Path | Conteúdo |
|---|---|
| `models/qwen3-legacy-doc-qlora-php-js-sql/` | adapter LoRA |
| `models/export/qwen2.5-1.5b-celx/` | pacote exportado |
| `outputs/eval_php_js_sql/` | avaliação |

No Jupyter: compacte e faça download, ou:

```bash
cd /workspace/Celx
tar -czf /workspace/celx-adapter.tgz models/export/qwen2.5-1.5b-celx
# scp de volta para o Mac
```

## 6) Custo e disco

- GPU ~**US\$ 0,25/h** (A4500) → stage A ~300k costuma sair na ordem de **US\$ 5–15**
- Pare o pod ao terminar (**Stop** / **Terminate**)
- Cache HF fica em `Celx/.hf_cache` (apague se o disco apertar)

## SSH (opcional)

```bash
ssh root@<IP> -p <PORTA> -i ~/.ssh/id_ed25519_runpod
```

Depois: `jupyter lab list` ou use só o Jupyter do painel.

## Troubleshooting

| Problema | O que fazer |
|---|---|
| `Projeto Celx não encontrado` | Zip em `/workspace/Celx-colab.zip` ou clone em `/workspace/Celx` |
| OOM CUDA | Baixe `per_device_train_batch_size` para 1 no yaml |
| Disco cheio | Apague `.hf_cache`, checkpoints extras (`save_total_limit: 1`) |
| Sem bitsandbytes | A célula de deps instala; reinicie o kernel se precisar |
