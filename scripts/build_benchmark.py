from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from legacy_doc.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrói benchmark com 100 casos por linguagem.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models-config", default="configs/model_candidates.yaml")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def stable_id(language: str, code: str) -> str:
    digest = hashlib.sha256(f"{language}\0{code}".encode()).hexdigest()[:16]
    return f"{language}-{digest}"


def select_unique(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["code"] in seen:
            continue
        seen.add(row["code"])
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    models_cfg = load_config(args.models_config)
    limit = models_cfg["benchmark"]["samples_per_language"]
    output = args.output or Path(models_cfg["benchmark"]["dataset"])
    seed = cfg["project"]["seed"]
    normalized_test = load_from_disk(str(Path(cfg["data"]["normalized_dir"]) / "test"))
    records: list[dict[str, Any]] = []

    for language in ("python", "php", "javascript"):
        subset = normalized_test.filter(lambda row: row["language"] == language).shuffle(seed=seed)
        candidates = [
            {
                "id": stable_id(language, row["code"]),
                "language": language,
                "code": row["code"],
                "source_reference": row["reference"],
                "source": "google/code_x_glue_ct_code_to_text",
                "expected_facts": [],
                "critical_traps": [],
            }
            for row in subset
        ]
        selected = select_unique(candidates, limit)
        if len(selected) < limit:
            raise ValueError(f"Apenas {len(selected)} casos únicos disponíveis para {language}")
        records.extend(selected)

    spider = load_dataset("xlangai/spider", split="validation").shuffle(seed=seed)
    sql_candidates = [
        {
            "id": stable_id("sql", row["query"]),
            "language": "sql",
            "code": row["query"],
            "source_reference": row["question"],
            "database_id": row["db_id"],
            "source": "xlangai/spider",
            "expected_facts": [],
            "critical_traps": [],
        }
        for row in spider
        if str(row["query"]).strip()
    ]
    selected_sql = select_unique(sql_candidates, limit)
    if len(selected_sql) < limit:
        raise ValueError(f"Apenas {len(selected_sql)} casos SQL únicos disponíveis")
    records.extend(selected_sql)

    counts = {language: 0 for language in ("python", "php", "javascript", "sql")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            counts[record["language"]] += 1
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "output": str(output),
        "total": len(records),
        "counts": counts,
        "seed": seed,
        "sources": {
            "python_php_javascript": "google/code_x_glue_ct_code_to_text",
            "sql": "xlangai/spider",
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
