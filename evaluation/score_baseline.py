from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = (
    "### Objetivo",
    "### Parâmetros",
    "### Retorno",
    "### Funcionamento",
    "### Regras de negócio identificadas",
    "### Pontos não determinados",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria planilha de avaliação cega do baseline.")
    parser.add_argument("--results", type=Path, default=Path("evaluation/results/baseline"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/human_scores.csv"))
    return parser.parse_args()


def read_results(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def main() -> None:
    args = parse_args()
    records = read_results(args.results)
    if not records:
        raise FileNotFoundError(f"Nenhum resultado JSONL encontrado em {args.results}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "id",
        "language",
        "structure_score",
        "elapsed_seconds",
        "fidelity_1_5",
        "coverage_1_5",
        "clarity_1_5",
        "business_rules_1_5",
        "human_structure_1_5",
        "critical_error",
        "notes",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            response = record["response"]
            present = sum(section in response for section in REQUIRED_SECTIONS)
            writer.writerow(
                {
                    "model": record["model"],
                    "id": record["id"],
                    "language": record["language"],
                    "structure_score": f"{present}/{len(REQUIRED_SECTIONS)}",
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
    print(f"Planilha criada em {args.output}")


if __name__ == "__main__":
    main()

