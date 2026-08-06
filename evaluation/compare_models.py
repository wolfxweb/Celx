from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
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
    parser = argparse.ArgumentParser(description="Resume resultados do benchmark por modelo.")
    parser.add_argument("--results", type=Path, default=Path("evaluation/results/baseline"))
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/results/model_comparison.csv")
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.results.glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        structure = [
            sum(section in record["response"] for section in REQUIRED_SECTIONS)
            / len(REQUIRED_SECTIONS)
            for record in records
        ]
        rows.append(
            {
                "model": records[0]["model"],
                "examples": len(records),
                "languages": ",".join(sorted({record["language"] for record in records})),
                "structure_percent": round(100 * mean(structure), 2),
                "average_seconds": round(mean(record["elapsed_seconds"] for record in records), 3),
                "average_generated_tokens": round(
                    mean(record["generated_tokens"] for record in records), 1
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(f"Nenhum resultado encontrado em {args.results}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparativo criado em {args.output}")


if __name__ == "__main__":
    main()

