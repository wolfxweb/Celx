from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from legacy_doc.config import load_config


TEMPLATE = """## Método ou função

Documentação smoke do exemplo `{name}`.

### Objetivo

Descrever o comportamento observável do trecho.

### Parâmetros

- Entradas presentes no código analisado.

### Retorno

Resultado produzido pelo trecho, se houver.

### Funcionamento

- Segue a sequência de instruções do código.

### Regras de negócio identificadas

- Apenas o comportamento explícito no código.

### Pontos não determinados

- Comportamento para entradas inválidas não descritas no trecho.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria curadoria + SFT mínimos para smoke de treino local."
    )
    parser.add_argument("--config", default="configs/train_local.yaml")
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=None,
        help="Default: data.curated_dir do config (train_local → dataset/curated_smoke).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    curated_dir = args.curated_dir or Path(cfg["data"]["curated_dir"])
    if curated_dir.resolve() == (Path.cwd() / "dataset" / "curated").resolve():
        raise SystemExit(
            "Recusando gravar smoke em dataset/curated/ (reservado a pt-BR real). "
            "Use dataset/curated_smoke/ (configs/train_local.yaml)."
        )
    examples_dir = Path("dataset/examples")
    template_row = json.loads(
        Path("dataset/curated/templates/exemplo_curado.jsonl").read_text(encoding="utf-8")
    )

    files = {
        "python": examples_dir / "calcula_total.py",
        "php": examples_dir / "calcula_total.php",
        "javascript": examples_dir / "calculaTotal.js",
        "sql": examples_dir / "pedidos_pendentes.sql",
    }
    rows: list[dict] = [template_row]
    for language, path in files.items():
        code = path.read_text(encoding="utf-8")
        rows.append(
            {
                "id": f"smoke-{language}-001",
                "language": language,
                "code": code,
                "documentation_pt": TEMPLATE.format(name=path.name),
                "source": "smoke_local",
                "review_status": "approved",
                "reviewer": "smoke",
                "review_notes": "Gerado para treino local.",
            }
        )

    train_rows = rows + [
        {**row, "id": f"{row['id']}-r{i}"} for i in range(2) for row in rows
    ]
    validation_rows = rows[:2]
    test_rows = rows[2:4]

    curated_dir.mkdir(parents=True, exist_ok=True)
    for split, payload in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        path = curated_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in payload:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(payload)} exemplos")

    subprocess.run(
        [sys.executable, "scripts/build_sft_dataset.py", "--config", args.config],
        check=True,
    )
    print("Smoke SFT pronto.")


if __name__ == "__main__":
    main()
