from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy_doc.prompts import SYSTEM_PROMPT

# Nome no Ollama / Continue (tag com iniciais).
OLLAMA_MODEL = "qwen2.5:1.5b-celx"
OLLAMA_BASE = "qwen2.5:1.5b"
# Pasta no disco (sem ":" — inválido/problemático em paths).
DEFAULT_OUTPUT = Path("models/export/qwen2.5-1.5b-celx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta o adapter LoRA treinado para uso (pacote + tag Ollama celx)."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/qwen3-legacy-doc-lora-real"),
        help="Pasta do adapter treinado (adapter_config.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Pasta de exportação.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Modelo-base Hugging Face (default: lido do adapter_config).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Mescla LoRA no modelo-base (pesado em RAM; pode falhar no M1 8GB).",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help=f"Tag Ollama (default: {OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Não roda `ollama create` (só grava Modelfile).",
    )
    return parser.parse_args()


def write_modelfile(out: Path, ollama_base: str) -> Path:
    path = out / "Modelfile"
    # Escapa aspas triplas no system prompt se houver.
    system = SYSTEM_PROMPT.replace('"""', "'''")
    path.write_text(
        f'FROM {ollama_base}\n\nSYSTEM """{system}"""\n',
        encoding="utf-8",
    )
    return path


def create_ollama_model(modelfile: Path, ollama_model: str) -> bool:
    ollama = shutil.which("ollama")
    if not ollama:
        print("Ollama não encontrado no PATH — Modelfile salvo; crie depois com:")
        print(f"  ollama create {ollama_model} -f {modelfile}")
        return False
    print(f"Registrando no Ollama: {ollama_model}")
    result = subprocess.run(
        [ollama, "create", ollama_model, "-f", str(modelfile)],
        check=False,
    )
    if result.returncode != 0:
        print(
            f"Falha no ollama create (rc={result.returncode}). "
            f"App Ollama aberto? Depois: ollama create {ollama_model} -f {modelfile}"
        )
        return False
    print(f"Ollama OK: {ollama_model}")
    return True


def main() -> None:
    args = parse_args()
    adapter = args.adapter.resolve()
    config_path = adapter / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Adapter não encontrado em {adapter}. "
            "Conclua o treino real antes de exportar."
        )

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = args.base_model or cfg.get("base_model_name_or_path") or "Qwen/Qwen3-1.7B"
    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    adapter_out = out / "adapter"
    adapter_out.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
    ):
        src = adapter / name
        if src.exists():
            shutil.copy2(src, adapter_out / name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        src = adapter / name
        if src.exists():
            shutil.copy2(src, adapter_out / name)

    ollama_model = args.ollama_model
    modelfile = write_modelfile(out, OLLAMA_BASE)
    ollama_ok = False
    if not args.skip_ollama:
        ollama_ok = create_ollama_model(modelfile, ollama_model)

    meta = {
        "base_model": base_model,
        "adapter_dir": "adapter",
        "merged": False,
        "ollama_model": ollama_model,
        "ollama_base": OLLAMA_BASE,
        "ollama_registered": ollama_ok,
        "how_to_use": (
            "python scripts/document_code.py "
            "--file dataset/examples/calcula_total.py "
            "--language python "
            f"--adapter {adapter_out}"
        ),
        "how_to_use_ollama": f"ollama run {ollama_model}",
    }

    if args.merge:
        print(f"Mesclando {base_model} + {adapter} (pode demorar / usar muita RAM)...")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter))
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter))
        merged = model.merge_and_unload()
        merged_dir = out / "merged"
        merged_dir.mkdir()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        meta["merged"] = True
        meta["merged_dir"] = "merged"
        print("Modelo mesclado em", merged_dir)

    usage = f"""# Modelo exportado — Celx

Base HF: `{base_model}`
Adapter: `adapter/`
Ollama: `{ollama_model}` (FROM `{OLLAMA_BASE}` + system Celx)

## Ollama / Continue

```bash
ollama create {ollama_model} -f {modelfile}
ollama run {ollama_model}
```

No Continue: model `{ollama_model}`.

## Uso com adapter (Transformers)

```bash
source .venv/bin/activate
python scripts/document_code.py \\
  --file dataset/examples/calcula_total.py \\
  --language python \\
  --adapter {adapter_out}
```
"""
    if meta["merged"]:
        usage += """
## Modelo mesclado

Pasta: `merged/`

Pode carregar direto com Transformers / converter para GGUF depois.
"""

    (out / "README.md").write_text(usage, encoding="utf-8")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Export pronto: {out}")
    print(f"Ollama tag: {ollama_model}")
    print(meta["how_to_use"])


if __name__ == "__main__":
    main()
