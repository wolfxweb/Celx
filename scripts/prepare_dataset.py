from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset
from legacy_doc.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara o CodeXGLUE para SFT.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help="Sobrescreve o tokenizer da configuração")
    return parser.parse_args()


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            if isinstance(value, list):
                return " ".join(str(item) for item in value)
            return str(value)
    return ""


def normalize(row: dict[str, Any], language: str) -> dict[str, Any]:
    code = first_present(row, ("code", "original_string", "function"))
    reference = first_present(row, ("docstring", "documentation", "summary"))
    fingerprint = hashlib.sha256(f"{language}\0{code}".encode()).hexdigest()
    return {
        "id": fingerprint,
        "language": language,
        "code": code.strip(),
        "reference": reference.strip(),
    }


def valid(row: dict[str, Any], minimum: int, maximum: int) -> bool:
    return minimum <= len(row["code"]) <= maximum and len(row["reference"]) >= 8


def prepare_split(
    dataset_name: str,
    split: str,
    languages: list[str],
    limit: int,
    data_cfg: dict[str, Any],
    seed: int,
    excluded_ids: set[str],
) -> Dataset:
    portions: list[Dataset] = []
    for language in languages:
        source = load_dataset(dataset_name, language, split=split)
        source = source.map(lambda row: normalize(row, language), remove_columns=source.column_names)
        source = source.filter(
            lambda row: valid(row, data_cfg["min_code_chars"], data_cfg["max_code_chars"])
        )
        source = source.shuffle(seed=seed).select(range(min(limit, len(source))))
        portions.append(source)
    merged = concatenate_datasets(portions).shuffle(seed=seed)
    selected: list[int] = []
    local_ids: set[str] = set()
    for index, fingerprint in enumerate(merged["id"]):
        if fingerprint not in excluded_ids and fingerprint not in local_ids:
            selected.append(index)
            local_ids.add(fingerprint)
    excluded_ids.update(local_ids)
    return merged.select(selected)


def load_sql_split(sql_dir: Path, split: str) -> Dataset | None:
    path = sql_dir / f"{split}.jsonl"
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "id": row["id"],
                    "language": "sql",
                    "code": str(row["code"]).strip(),
                    "reference": str(row["reference"]).strip(),
                }
            )
    return Dataset.from_list(rows) if rows else None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    random.seed(cfg["project"]["seed"])
    model_name = args.model or cfg["model"]["name"]
    data_cfg = cfg["data"]
    output = Path(data_cfg["normalized_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": data_cfg["dataset_name"],
        "model_for_future_tokenization": model_name,
        "status": "normalized_not_ready_for_sft",
        "languages": list(data_cfg.get("codexglue_languages", [])),
        "sql_merged": False,
    }
    seen_ids: set[str] = set()
    prepared_splits: dict[str, Dataset] = {}
    # Reserva exemplos de teste e validação antes do treino para evitar vazamento.
    for split in ("test", "validation", "train"):
        limit = data_cfg[f"max_{split}_samples_per_language"]
        prepared = prepare_split(
            data_cfg["dataset_name"],
            split,
            data_cfg["codexglue_languages"],
            limit,
            data_cfg,
            cfg["project"]["seed"],
            seen_ids,
        )
        prepared_splits[split] = prepared

    sql_dir = Path(data_cfg.get("sql_dataset_path", "dataset/sql"))
    want_sql = "sql" in [str(x).lower() for x in data_cfg.get("target_languages", [])]
    if want_sql:
        sql_parts: list[str] = []
        for split in ("train", "validation", "test"):
            sql_ds = load_sql_split(sql_dir, split)
            if sql_ds is None:
                continue
            # Evita IDs já usados pelo CodeXGLUE (improvável, mas seguro).
            keep = [
                i
                for i, item_id in enumerate(sql_ds["id"])
                if item_id not in seen_ids
            ]
            if not keep:
                continue
            sql_ds = sql_ds.select(keep)
            seen_ids.update(sql_ds["id"])
            prepared_splits[split] = concatenate_datasets(
                [prepared_splits[split], sql_ds]
            ).shuffle(seed=cfg["project"]["seed"])
            sql_parts.append(f"{split}:{len(sql_ds)}")
        if sql_parts:
            manifest["sql_merged"] = True
            manifest["sql_source"] = str(sql_dir)
            manifest["sql_counts"] = sql_parts
            manifest["languages"] = list(
                dict.fromkeys([*manifest["languages"], "sql"])
            )
        else:
            print(
                f"Aviso: SQL pedido em target_languages, mas sem JSONL em {sql_dir}. "
                "Rode scripts/prepare_sql_spider.py antes."
            )

    for split in ("train", "validation", "test"):
        prepared = prepared_splits[split]
        prepared.save_to_disk(str(output / split))
        prepared.to_json(output / f"{split}.jsonl", force_ascii=False)
        languages = sorted(set(prepared["language"]))
        manifest[split] = {
            "rows": len(prepared),
            "path": str(output / split),
            "languages": languages,
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
