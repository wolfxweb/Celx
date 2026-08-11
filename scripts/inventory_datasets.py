from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventaria camadas de dados (EN vs pt-BR) para decidir onde treinar."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/curated/INVENTARIO.md"),
        help="Markdown com o inventário (também grava .json ao lado).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_arrow(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        from datasets import load_from_disk
    except ImportError:
        return {"error": "datasets não instalado", "path": str(path)}
    ds = load_from_disk(str(path))
    langs = Counter(ds["language"]) if "language" in ds.column_names else {}
    return {"rows": len(ds), "languages": dict(sorted(langs.items())), "path": str(path)}


def summarize_jsonl_dir(directory: Path, splits: tuple[str, ...] = ("train", "validation", "test")) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(directory), "exists": directory.exists(), "splits": {}}
    if not directory.exists():
        return report
    total = 0
    langs: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    with_pt = 0
    for split in splits:
        rows = read_jsonl(directory / f"{split}.jsonl")
        split_langs = Counter(str(r.get("language", "?")) for r in rows)
        split_status = Counter(str(r.get("review_status", "?")) for r in rows)
        split_src = Counter(str(r.get("source", "?")) for r in rows)
        pt = sum(1 for r in rows if str(r.get("documentation_pt", "")).strip())
        report["splits"][split] = {
            "rows": len(rows),
            "languages": dict(sorted(split_langs.items())),
            "review_status": dict(sorted(split_status.items())),
            "sources": dict(sorted(split_src.items())),
            "with_documentation_pt": pt,
        }
        total += len(rows)
        langs.update(split_langs)
        sources.update(split_src)
        statuses.update(split_status)
        with_pt += pt
    report["total_rows"] = total
    report["languages"] = dict(sorted(langs.items()))
    report["sources"] = dict(sorted(sources.items()))
    report["review_status"] = dict(sorted(statuses.items()))
    report["with_documentation_pt"] = with_pt
    return report


def summarize_queues(directory: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(directory), "exists": directory.exists(), "splits": {}}
    if not directory.exists():
        return report
    total = 0
    pending = 0
    langs: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        path = directory / f"{split}_review_queue.jsonl"
        rows = read_jsonl(path)
        split_langs = Counter(str(r.get("language", "?")) for r in rows)
        split_pending = sum(1 for r in rows if r.get("review_status") == "pending")
        report["splits"][split] = {
            "rows": len(rows),
            "pending": split_pending,
            "languages": dict(sorted(split_langs.items())),
            "path": str(path) if path.exists() else None,
        }
        total += len(rows)
        pending += split_pending
        langs.update(split_langs)
    report["total_rows"] = total
    report["pending"] = pending
    report["languages"] = dict(sorted(langs.items()))
    manifest = directory / "manifest.json"
    if manifest.exists():
        report["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    return report


def where_to_train(pt_approved: int, queue_pending: int, en_local: int) -> list[str]:
    lines = [
        "## Onde treinar (orientação)",
        "",
        "| Volume pt-BR aprovado | Onde | Motivo |",
        "|---:|---|---|",
        "| ≤ 50 (smoke) | M1 local | Só valida pipeline |",
        "| 200–800 (piloto) | M1 local | Cabe em LoRA/MPS; qualidade real começa aqui |",
        "| 800–2.400 (meta v1) | M1 ou nuvem | M1 lento; RunPod/Kaggle se quiser 1 época rápida |",
        "| 8.000 (escala 2k/lang) | Nuvem QLoRA | Fora do conforto do M1 8GB |",
        "",
        f"- **pt-BR aprovado agora:** {pt_approved}",
        f"- **fila pendente (a curar):** {queue_pending}",
        f"- **EN embrulhado (CodeXGLUE local, se existir):** {en_local}",
        "",
    ]
    if pt_approved < 50:
        lines.append(
            "**Decisão sugerida:** continue no M1 com o notebook 04 (EN embrulhado) "
            "só para pipeline; em paralelo cure o piloto pt-BR (50/10/10 por linguagem)."
        )
    elif pt_approved < 800:
        lines.append(
            "**Decisão sugerida:** treine pt-BR no M1; nuvem só se o tempo local ficar "
            "inaceitável (>~1 dia)."
        )
    else:
        lines.append(
            "**Decisão sugerida:** meta v1 atingida — treino no M1 ainda possível; "
            "RunPod/Kaggle se quiser iterar mais rápido."
        )
    lines.append("")
    return lines


def md_section(title: str, body: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", "", f"- Path: `{body.get('path', '?')}`"]
    if not body.get("exists", True) and "exists" in body:
        lines.extend(["- Status: **ausente**", ""])
        return lines
    if "total_rows" in body:
        lines.append(f"- Total: **{body['total_rows']}**")
    if "pending" in body:
        lines.append(f"- Pendentes: **{body['pending']}**")
    if "with_documentation_pt" in body:
        lines.append(f"- Com `documentation_pt`: **{body['with_documentation_pt']}**")
    if body.get("languages"):
        lines.append(f"- Linguagens: `{body['languages']}`")
    if body.get("sources"):
        lines.append(f"- Fontes: `{body['sources']}`")
    if body.get("review_status"):
        lines.append(f"- Status: `{body['review_status']}`")
    if body.get("splits"):
        lines.append("")
        for split, info in body["splits"].items():
            if isinstance(info, dict) and "rows" in info:
                extra = ""
                if "pending" in info:
                    extra = f", pending={info['pending']}"
                if "languages" in info:
                    extra += f", langs={info['languages']}"
                lines.append(f"- `{split}`: {info['rows']}{extra}")
            elif isinstance(info, dict) and "rows" in info:
                lines.append(f"- `{split}`: {info}")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    processed = ROOT / "dataset" / "processed"

    curated = summarize_jsonl_dir(ROOT / "dataset" / "curated")
    smoke = summarize_jsonl_dir(ROOT / "dataset" / "curated_smoke")
    queues = summarize_queues(ROOT / "dataset" / "curated" / "queues")

    layers_en: dict[str, Any] = {}
    for name in ("codexglue", "full"):
        layer: dict[str, Any] = {"path": str(processed / name), "splits": {}}
        total = 0
        langs: Counter[str] = Counter()
        for split in ("train", "validation", "test"):
            info = count_arrow(processed / name / split)
            layer["splits"][split] = info
            if info and "rows" in info:
                total += int(info["rows"])
                langs.update(info.get("languages") or {})
        layer["total_rows"] = total
        layer["languages"] = dict(sorted(langs.items()))
        layer["exists"] = (processed / name).exists()
        layers_en[name] = layer

    sft_layers: dict[str, Any] = {}
    for name in ("sft", "sft_real", "sft_full"):
        layer = {"path": str(processed / name), "splits": {}, "exists": (processed / name).exists()}
        total = 0
        for split in ("train", "validation", "test"):
            info = count_arrow(processed / name / split)
            layer["splits"][split] = info
            if info and "rows" in info:
                total += int(info["rows"])
        layer["total_rows"] = total
        manifest = processed / name / "manifest.json"
        if manifest.exists():
            layer["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        sft_layers[name] = layer

    en_train = 0
    for key in ("full", "codexglue"):
        split = layers_en[key]["splits"].get("train")
        if split and "rows" in split:
            en_train = int(split["rows"])
            break

    inventory: dict[str, Any] = {
        "kind": "celx_dataset_inventory",
        "note": (
            "pt-BR real = documentation_pt curada/aprovada. "
            "EN embrulhado = reference CodeXGLUE/Spider em template PT (não é documentação pt-BR)."
        ),
        "targets": {
            "piloto_per_language": {"train": 50, "validation": 10, "test": 10, "total": 280},
            "meta_v1_per_language": {"train": 500, "validation": 50, "test": 50, "total": 2400},
            "escala_2k_per_language": {
                "train": 1600,
                "validation": 200,
                "test": 200,
                "total_per_language": 2000,
                "total": 8000,
            },
        },
        "pt_br": {
            "curated_approved": curated,
            "curated_smoke": smoke,
            "review_queues": queues,
        },
        "english_normalized": layers_en,
        "sft": sft_layers,
        "decision_inputs": {
            "pt_br_approved_rows": curated.get("with_documentation_pt", 0),
            "queue_pending_rows": queues.get("pending", 0),
            "en_normalized_train_rows": en_train,
        },
    }

    md: list[str] = [
        "# Inventário de dados Celx",
        "",
        "Gerado por `python scripts/inventory_datasets.py`.",
        "",
        inventory["note"],
        "",
        "## Metas pt-BR",
        "",
        "| Cenário | Train/lang | Val/lang | Test/lang | Total (4 langs) | Onde treinar |",
        "|---|---:|---:|---:|---:|---|",
        "| Smoke | ~4 | ~1 | ~1 | ~19 | M1 (pipeline) |",
        "| Piloto | 50 | 10 | 10 | **280** | M1 |",
        "| Meta v1 | 500 | 50 | 50 | **2.400** | M1 ou nuvem |",
        "| Escala 2k | 1600 | 200 | 200 | **8.000** (2k/lang) | Nuvem QLoRA |",
        "| EN cheio PHP+JS+SQL | — | — | — | **~300k** train (sem Python) | Nuvem A4500+ |",
        "",
        "Nota: ~500k era Python+PHP+JS. Sem Python ficam ~241k PHP + ~58k JS + ~7k SQL.",
        "",
    ]
    md.extend(
        where_to_train(
            int(inventory["decision_inputs"]["pt_br_approved_rows"]),
            int(inventory["decision_inputs"]["queue_pending_rows"]),
            int(inventory["decision_inputs"]["en_normalized_train_rows"]),
        )
    )
    md.extend(md_section("pt-BR aprovado (`dataset/curated`)", curated))
    md.extend(md_section("Smoke pt-BR (`dataset/curated_smoke`)", smoke))
    md.extend(md_section("Filas de curadoria (`dataset/curated/queues`)", queues))
    for name, layer in layers_en.items():
        md.extend(md_section(f"Normalizado EN (`dataset/processed/{name}`)", layer))
    for name, layer in sft_layers.items():
        kind = ""
        if layer.get("manifest"):
            kind = f" — kind=`{layer['manifest'].get('kind', '?')}`"
        md.extend(md_section(f"SFT (`dataset/processed/{name}`){kind}", layer))

    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(json_path)
    print(
        json.dumps(
            {
                "pt_br_approved": inventory["decision_inputs"]["pt_br_approved_rows"],
                "queue_pending": inventory["decision_inputs"]["queue_pending_rows"],
                "en_train": inventory["decision_inputs"]["en_normalized_train_rows"],
                "targets": inventory["targets"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
