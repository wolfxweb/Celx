from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import load_from_disk

from legacy_doc.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa o CodeXGLUE normalizado.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_analysis"))
    return parser.parse_args()


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(ordered),
        "mean": round(mean(ordered), 2),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = Path(cfg["data"]["normalized_dir"])
    report: dict[str, Any] = {"source": str(data_dir), "splits": {}}
    markdown = ["# Análise do CodeXGLUE normalizado", ""]
    all_ids: Counter[str] = Counter()

    for split in ("train", "validation", "test"):
        dataset = load_from_disk(str(data_dir / split))
        language_counts = Counter(dataset["language"])
        all_ids.update(dataset["id"])
        split_report = {
            "rows": len(dataset),
            "languages": dict(sorted(language_counts.items())),
            "code_chars": distribution([len(value) for value in dataset["code"]]),
            "reference_chars": distribution([len(value) for value in dataset["reference"]]),
        }
        report["splits"][split] = split_report
        markdown.extend(
            [
                f"## {split}",
                "",
                f"- Linhas: {len(dataset)}",
                f"- Linguagens: {dict(sorted(language_counts.items()))}",
                f"- Código (caracteres): {split_report['code_chars']}",
                f"- Referência (caracteres): {split_report['reference_chars']}",
                "",
            ]
        )

    duplicates = sum(count - 1 for count in all_ids.values() if count > 1)
    report["duplicate_ids_across_splits"] = duplicates
    markdown.extend(["## Integridade", "", f"- IDs duplicados entre divisões: {duplicates}", ""])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "analysis.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

