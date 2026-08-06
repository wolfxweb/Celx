from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from legacy_doc.config import load_config


LANGUAGES = ("python", "php", "javascript", "sql")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria filas balanceadas para curadoria humana.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/curated/queues"))
    parser.add_argument("--train-per-language", type=int, default=500)
    parser.add_argument("--validation-per-language", type=int, default=50)
    parser.add_argument("--test-per-language", type=int, default=50)
    parser.add_argument("--benchmark", type=Path, default=Path("dataset/benchmark/expanded_100.jsonl"))
    return parser.parse_args()


def fingerprint(language: str, code: str) -> str:
    return hashlib.sha256(f"{language}\0{code.strip()}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def queue_record(
    language: str, code: str, reference: str, source: str, split: str
) -> dict[str, Any]:
    item_id = fingerprint(language, code)
    return {
        "id": item_id,
        "language": language,
        "code": code.strip(),
        "source_reference": reference.strip(),
        "documentation_pt": "",
        "source": source,
        "split": split,
        "review_status": "pending",
        "reviewer": "",
        "review_notes": "",
    }


def take_unique(
    candidates: list[dict[str, Any]], limit: int, excluded: set[str]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in candidates:
        item_id = fingerprint(row["language"], row["code"])
        if item_id in excluded:
            continue
        excluded.add(item_id)
        selected.append(row)
        if len(selected) == limit:
            return selected
    raise ValueError(f"Somente {len(selected)} registros únicos disponíveis; esperado: {limit}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    normalized_dir = Path(cfg["data"]["normalized_dir"])
    counts = {
        "train": args.train_per_language,
        "validation": args.validation_per_language,
        "test": args.test_per_language,
    }

    excluded = {
        fingerprint(row["language"], row["code"])
        for row in read_jsonl(args.benchmark)
        if row.get("language") and row.get("code")
    }
    benchmark_exclusions = len(excluded)
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in counts}

    for split in ("test", "validation", "train"):
        dataset = load_from_disk(str(normalized_dir / split))
        for language in LANGUAGES[:-1]:
            subset = dataset.filter(lambda row, lang=language: row["language"] == lang).shuffle(
                seed=seed
            )
            candidates = [
                queue_record(
                    language,
                    row["code"],
                    row["reference"],
                    "google/code_x_glue_ct_code_to_text",
                    split,
                )
                for row in subset
            ]
            output[split].extend(take_unique(candidates, counts[split], excluded))

    sql = load_dataset("xlangai/spider", split="train").shuffle(seed=seed)
    sql_candidates = [
        queue_record("sql", row["query"], row["question"], "xlangai/spider", "")
        for row in sql
        if str(row["query"]).strip() and str(row["question"]).strip()
    ]
    offset = 0
    for split in ("test", "validation", "train"):
        needed = counts[split]
        selected: list[dict[str, Any]] = []
        while offset < len(sql_candidates) and len(selected) < needed:
            row = sql_candidates[offset]
            offset += 1
            item_id = fingerprint("sql", row["code"])
            if item_id in excluded:
                continue
            excluded.add(item_id)
            row["split"] = split
            selected.append(row)
        if len(selected) != needed:
            raise ValueError(f"SQL/{split}: {len(selected)} disponíveis; esperado: {needed}")
        output[split].extend(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "pending_human_review",
        "seed": seed,
        "benchmark_fingerprints_excluded": benchmark_exclusions,
        "splits": {},
    }
    for split, rows in output.items():
        rows.sort(key=lambda row: (row["language"], row["id"]))
        path = args.output_dir / f"{split}_review_queue.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        per_language = {lang: sum(row["language"] == lang for row in rows) for lang in LANGUAGES}
        manifest["splits"][split] = {
            "path": str(path),
            "rows": len(rows),
            "per_language": per_language,
        }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
