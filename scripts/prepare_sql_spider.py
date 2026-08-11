from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from legacy_doc.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa Spider e gera dataset/sql/{train,validation,test}.jsonl."
    )
    parser.add_argument("--config", default="configs/train_full.yaml")
    return parser.parse_args()


def fingerprint(code: str) -> str:
    return hashlib.sha256(f"sql\0{code.strip()}".encode()).hexdigest()


def to_row(query: str, question: str) -> dict[str, Any] | None:
    code = str(query).strip()
    reference = str(question).strip()
    if len(code) < 8 or len(reference) < 8:
        return None
    return {
        "id": fingerprint(code),
        "language": "sql",
        "code": code,
        "reference": reference,
        "source": "xlangai/spider",
        "license": "CC-BY-SA-4.0",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    seed = int(cfg["project"]["seed"])
    random.seed(seed)

    out = Path(data_cfg.get("sql_dataset_path", "dataset/sql"))
    out.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": int(data_cfg["max_train_samples_per_language"]),
        "validation": int(data_cfg["max_validation_samples_per_language"]),
        "test": int(data_cfg["max_test_samples_per_language"]),
    }
    needed = sum(limits.values())

    print("Baixando xlangai/spider (split train)...")
    spider = load_dataset("xlangai/spider", split="train")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in spider.shuffle(seed=seed):
        item = to_row(row["query"], row["question"])
        if item is None or item["id"] in seen:
            continue
        seen.add(item["id"])
        candidates.append(item)
        if len(candidates) >= needed + 50:
            break

    # Spider tem ~7k; se o teto do config for "completo" (ex. 300k), usa tudo.
    if len(candidates) < needed:
        n = len(candidates)
        print(
            f"Aviso: Spider rendeu {n} únicos (pedido {needed}). "
            "Distribuindo o disponível em test/val/train."
        )
        n_test = min(limits["test"], max(1, n // 10))
        n_val = min(limits["validation"], max(1, n // 10))
        # Garante que sobre a maior parte para train.
        while n_test + n_val >= n and (n_test > 1 or n_val > 1):
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        n_train = n - n_test - n_val
        if n_train < 1:
            raise RuntimeError(f"Spider insuficiente para splits ({n} únicos).")
        limits = {"test": n_test, "validation": n_val, "train": n_train}

    # Reserva test → validation → train (sem vazamento entre splits).
    offset = 0
    manifest: dict[str, Any] = {
        "source": "xlangai/spider",
        "seed": seed,
        "splits": {},
    }
    for split in ("test", "validation", "train"):
        n = limits[split]
        chunk = candidates[offset : offset + n]
        offset += n
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for item in chunk:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"rows": len(chunk), "path": str(path)}
        print(f"SQL {split}: {len(chunk)} → {path}")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
