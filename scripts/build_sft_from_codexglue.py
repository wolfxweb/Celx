from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from legacy_doc.config import load_config
from legacy_doc.prompts import training_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrói SFT a partir do CodeXGLUE normalizado (treino real, etapa 1)."
    )
    parser.add_argument("--config", default="configs/train_real.yaml")
    parser.add_argument(
        "--doc-language",
        default=None,
        help="Idioma do template SFT (default: pt-BR; use en no stage bilingue híbrido).",
    )
    return parser.parse_args()


def render(
    row: dict[str, Any], tokenizer: AutoTokenizer, doc_language: str
) -> dict[str, Any]:
    messages = training_messages(
        row["language"], row["code"], row["reference"], doc_language=doc_language
    )
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        # Dados de treino: resposta final sem bloco think (inferência liga thinking).
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return {
        "id": row["id"],
        "language": row["language"],
        "doc_language": doc_language,
        "code": row["code"],
        "reference": row["reference"],
        "messages": messages,
        "text": text,
        "source": "code_x_glue",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    normalized = Path(data_cfg["normalized_dir"])
    output_dir = Path(data_cfg["sft_dir"])
    min_train = int(cfg.get("training", {}).get("min_train_samples", 100))
    doc_language = args.doc_language or cfg.get("generation", {}).get("doc_language", "pt-BR")

    if not (normalized / "train").exists():
        raise FileNotFoundError(
            f"CodeXGLUE normalizado ausente em {normalized}. "
            "Rode antes: python scripts/prepare_dataset.py --config configs/train_real.yaml"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model": cfg["model"]["name"],
        "source": "normalized_codexglue_plus_sql",
        "normalized_dir": str(normalized),
        "kind": "full_pipeline",
        "doc_language": doc_language,
        "splits": {},
        "languages": {},
    }
    for split in ("train", "validation", "test"):
        source = load_from_disk(str(normalized / split))
        rendered = [render(row, tokenizer, doc_language) for row in source]
        dataset = Dataset.from_list(rendered)
        dataset.save_to_disk(str(output_dir / split))
        dataset.to_json(output_dir / f"{split}.jsonl", force_ascii=False)
        langs: dict[str, int] = {}
        for lang in dataset["language"]:
            langs[lang] = langs.get(lang, 0) + 1
        manifest["splits"][split] = len(dataset)
        manifest["languages"][split] = dict(sorted(langs.items()))
        print(f"{split}: {len(dataset)} exemplos {langs} → {output_dir / split}")

    if manifest["splits"]["train"] < min_train:
        raise RuntimeError(
            f"Train tem só {manifest['splits']['train']} exemplos "
            f"(mínimo {min_train}). Isso parece smoke, não treino real."
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
