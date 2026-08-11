from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from legacy_doc.config import load_config
from legacy_doc.prompts import curated_messages, required_sections


REQUIRED_SECTIONS = required_sections("pt-BR")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida a curadoria e constrói o dataset SFT.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(row: dict[str, Any], split: str, line: int) -> None:
    required = ("language", "code", "documentation_pt")
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"{split}:{line}: campos ausentes: {', '.join(missing)}")
    if row["language"] not in {"python", "php", "javascript", "sql"}:
        raise ValueError(f"{split}:{line}: linguagem inválida: {row['language']}")
    if row.get("review_status") != "approved":
        raise ValueError(f"{split}:{line}: exemplo ainda não aprovado por revisão humana")
    absent_sections = [
        section for section in REQUIRED_SECTIONS if section not in row["documentation_pt"]
    ]
    if absent_sections:
        raise ValueError(f"{split}:{line}: seções ausentes: {', '.join(absent_sections)}")


def fingerprint(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{row['language']}\0{row['code'].strip()}".encode()).hexdigest()


def render(row: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, Any]:
    messages = curated_messages(row["language"], row["code"], row["documentation_pt"], "pt-BR")
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return {**row, "id": row.get("id") or fingerprint(row), "messages": messages, "text": text}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    curated_dir = Path(data_cfg["curated_dir"])
    output_dir = Path(data_cfg["sft_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path("dataset/benchmark/expanded_100.jsonl")
    benchmark_ids: set[str] = set()
    if benchmark_path.exists():
        for row in read_jsonl(benchmark_path):
            if row.get("language") and row.get("code"):
                benchmark_ids.add(fingerprint(row))

    seen: set[str] = set()
    manifest: dict[str, Any] = {"model": cfg["model"]["name"], "splits": {}}
    for split in ("test", "validation", "train"):
        source = curated_dir / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Curadoria ausente: {source}")
        rows = read_jsonl(source)
        rendered: list[dict[str, Any]] = []
        for line, row in enumerate(rows, start=1):
            validate(row, split, line)
            item_id = fingerprint(row)
            if item_id in benchmark_ids:
                raise ValueError(f"{split}:{line}: exemplo também está no benchmark")
            if item_id in seen:
                continue
            seen.add(item_id)
            rendered.append(render(row, tokenizer))
        dataset = Dataset.from_list(rendered)
        dataset.save_to_disk(str(output_dir / split))
        dataset.to_json(output_dir / f"{split}.jsonl", force_ascii=False)
        manifest["splits"][split] = len(dataset)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
