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
    parser.add_argument("--config", default="configs/curadoria.yaml")
    parser.add_argument(
        "--mode",
        choices=("piloto", "meta_v1", "escala_2k"),
        default=None,
        help="Volumes de curadoria (default: curation.mode no YAML).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-per-language", type=int, default=None)
    parser.add_argument("--validation-per-language", type=int, default=None)
    parser.add_argument("--test-per-language", type=int, default=None)
    parser.add_argument("--benchmark", type=Path, default=None)
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


def resolve_counts(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[str, dict[str, int], Path, Path]:
    curation = cfg.get("curation") or {}
    mode = args.mode or curation.get("mode") or "piloto"
    preset = dict(curation.get(mode) or {})
    counts = {
        "train": args.train_per_language
        if args.train_per_language is not None
        else int(preset.get("train_per_language", 50)),
        "validation": args.validation_per_language
        if args.validation_per_language is not None
        else int(preset.get("validation_per_language", 10)),
        "test": args.test_per_language
        if args.test_per_language is not None
        else int(preset.get("test_per_language", 10)),
    }
    output_dir = args.output_dir or Path(curation.get("queue_dir", "dataset/curated/queues"))
    benchmark = args.benchmark or Path(
        curation.get("benchmark", "dataset/benchmark/expanded_100.jsonl")
    )
    return mode, counts, output_dir, benchmark


def resolve_normalized_dir(cfg: dict[str, Any]) -> Path:
    preferred = Path(cfg["data"]["normalized_dir"])
    if (preferred / "train").exists():
        return preferred
    fallback = Path("dataset/processed/codexglue")
    if (fallback / "train").exists():
        print(f"Aviso: {preferred} ausente — usando {fallback}")
        return fallback
    raise FileNotFoundError(
        f"Normalized dir não encontrado: {preferred} (nem fallback {fallback}). "
        "Rode prepare_dataset.py / prepare_sql_spider.py antes."
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    mode, counts, output_dir, benchmark = resolve_counts(args, cfg)
    normalized_dir = resolve_normalized_dir(cfg)

    excluded = {
        fingerprint(row["language"], row["code"])
        for row in read_jsonl(benchmark)
        if row.get("language") and row.get("code")
    }
    benchmark_exclusions = len(excluded)
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in counts}

    for split in ("test", "validation", "train"):
        dataset = load_from_disk(str(normalized_dir / split))
        available_langs = set(dataset["language"])
        for language in LANGUAGES[:-1]:
            if language not in available_langs:
                raise ValueError(
                    f"{normalized_dir}/{split} sem linguagem '{language}'. "
                    "Reprepare o pool normalizado."
                )
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

    print("Baixando/carregando xlangai/spider para SQL na fila...")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(rows) for rows in output.values())
    manifest: dict[str, Any] = {
        "status": "pending_human_review",
        "language": "pt-BR",
        "mode": mode,
        "seed": seed,
        "normalized_dir": str(normalized_dir),
        "benchmark_fingerprints_excluded": benchmark_exclusions,
        "totals": {
            "rows": total,
            "per_language_target": counts,
            "languages": list(LANGUAGES),
        },
        "splits": {},
    }
    for split, rows in output.items():
        rows.sort(key=lambda row: (row["language"], row["id"]))
        path = output_dir / f"{split}_review_queue.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        per_language = {lang: sum(row["language"] == lang for row in rows) for lang in LANGUAGES}
        manifest["splits"][split] = {
            "path": str(path),
            "rows": len(rows),
            "per_language": per_language,
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
