from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from legacy_doc.config import load_config
from legacy_doc.prompts import (
    curated_messages,
    documentation_field_for,
    normalize_doc_language,
    required_sections,
    training_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monta SFT bilingue: 2 conversas por código (pt-BR + en) quando houver docs; "
            "no CodeXGLUE gera o par EN estruturado e, se houver curadoria, o par pt-BR."
        )
    )
    parser.add_argument("--config", default="configs/train_bilingual.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(language: str, code: str) -> str:
    return hashlib.sha256(f"{language}\0{code.strip()}".encode()).hexdigest()


def apply_template(messages: list[dict[str, str]], tokenizer: AutoTokenizer) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_pair(
    *,
    item_id: str,
    language: str,
    code: str,
    doc_language: str,
    documentation: str,
    tokenizer: AutoTokenizer,
    source: str,
    reference: str = "",
) -> dict[str, Any]:
    doc_language = normalize_doc_language(doc_language)
    messages = curated_messages(language, code, documentation, doc_language)
    return {
        "id": f"{item_id}::{doc_language}",
        "base_id": item_id,
        "language": language,
        "doc_language": doc_language,
        "code": code,
        "reference": reference,
        "documentation": documentation.strip(),
        "messages": messages,
        "text": apply_template(messages, tokenizer),
        "source": source,
    }


def validate_doc(text: str, doc_language: str, where: str) -> None:
    missing = [s for s in required_sections(doc_language) if s not in text]
    if missing:
        raise ValueError(f"{where}: seções ausentes ({doc_language}): {', '.join(missing)}")


def from_curated(
    curated_dir: Path,
    doc_languages: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in ("train", "validation", "test")}
    for split in out:
        path = curated_dir / f"{split}.jsonl"
        for line, row in enumerate(read_jsonl(path), start=1):
            if row.get("review_status") and row.get("review_status") != "approved":
                continue
            language = str(row["language"])
            code = str(row["code"])
            base_id = str(row.get("id") or fingerprint(language, code))
            for doc_lang in doc_languages:
                field = documentation_field_for(doc_lang)
                doc = str(row.get(field, "")).strip()
                # Fallback EN: source_reference embrulhado só se não houver documentation_en.
                if not doc and doc_lang == "en" and str(row.get("source_reference", "")).strip():
                    messages = training_messages(
                        language, code, str(row["source_reference"]), doc_language="en"
                    )
                    doc = messages[-1]["content"]
                    source = "curated_en_from_reference"
                elif not doc:
                    continue
                else:
                    source = "curated_bilingual"
                    validate_doc(doc, doc_lang, f"{split}:{line}:{field}")
                out[split].append(
                    render_pair(
                        item_id=base_id,
                        language=language,
                        code=code,
                        doc_language=doc_lang,
                        documentation=doc,
                        tokenizer=tokenizer,
                        source=source,
                        reference=str(row.get("source_reference", "")),
                    )
                )
    return out


def from_normalized(
    normalized_dir: Path,
    doc_languages: list[str],
    tokenizer: AutoTokenizer,
    curated_pt_by_id: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """EN a partir da reference; pt-BR só se o mesmo código estiver curado."""
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in ("train", "validation", "test")}
    want_en = "en" in doc_languages
    want_pt = "pt-BR" in doc_languages
    for split in out:
        path = normalized_dir / split
        if not path.exists():
            continue
        ds = load_from_disk(str(path))
        for row in ds:
            language = row["language"]
            code = row["code"]
            base_id = row["id"]
            reference = row.get("reference", "")
            if want_en and str(reference).strip():
                messages = training_messages(language, code, reference, doc_language="en")
                out[split].append(
                    render_pair(
                        item_id=base_id,
                        language=language,
                        code=code,
                        doc_language="en",
                        documentation=messages[-1]["content"],
                        tokenizer=tokenizer,
                        source="codexglue_en",
                        reference=reference,
                    )
                )
            if want_pt and base_id in curated_pt_by_id:
                doc = curated_pt_by_id[base_id]
                validate_doc(doc, "pt-BR", f"normalized+curated:{base_id}")
                out[split].append(
                    render_pair(
                        item_id=base_id,
                        language=language,
                        code=code,
                        doc_language="pt-BR",
                        documentation=doc,
                        tokenizer=tokenizer,
                        source="curated_pt_on_normalized",
                        reference=reference,
                    )
                )
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    bilingual = cfg.get("bilingual") or {}
    doc_languages = [
        normalize_doc_language(x)
        for x in bilingual.get("doc_languages", ["pt-BR", "en"])
    ]
    mode = str(bilingual.get("mode", "curated_pairs"))
    # curated_pairs: só curated (2 docs quando existirem)
    # hybrid: curated + EN do CodeXGLUE normalizado

    curated_dir = Path(data_cfg["curated_dir"])
    output_dir = Path(data_cfg["sft_dir"])
    normalized_dir = Path(data_cfg.get("normalized_dir", "dataset/processed/full"))
    min_train = int(cfg.get("training", {}).get("min_train_samples", 1))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    output_dir.mkdir(parents=True, exist_ok=True)

    curated_map = from_curated(curated_dir, doc_languages, tokenizer)
    curated_pt_by_id: dict[str, str] = {}
    for split_rows in curated_map.values():
        for row in split_rows:
            if row["doc_language"] == "pt-BR":
                curated_pt_by_id[row["base_id"]] = row["documentation"]

    if mode == "hybrid":
        normalized_map = from_normalized(
            normalized_dir, doc_languages, tokenizer, curated_pt_by_id
        )
    else:
        normalized_map = {s: [] for s in ("train", "validation", "test")}

    manifest: dict[str, Any] = {
        "model": cfg["model"]["name"],
        "kind": "bilingual_sft",
        "mode": mode,
        "doc_languages": doc_languages,
        "splits": {},
        "by_doc_language": {},
        "note": (
            "2 respostas/código quando pt-BR e en existem. "
            "CodeXGLUE sozinho só gera EN estruturado; pt-BR exige curadoria."
        ),
    }

    for split in ("train", "validation", "test"):
        rows = curated_map[split] + normalized_map[split]
        # Dedup por id (base_id::doc_lang)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            unique.append(row)
        if not unique and split != "train":
            dataset = Dataset.from_list([])
        else:
            dataset = Dataset.from_list(unique)
        dataset.save_to_disk(str(output_dir / split))
        if len(dataset):
            dataset.to_json(output_dir / f"{split}.jsonl", force_ascii=False)
        counts: dict[str, int] = {}
        if len(dataset) and "doc_language" in dataset.column_names:
            for lang in dataset["doc_language"]:
                counts[lang] = counts.get(lang, 0) + 1
        manifest["splits"][split] = len(dataset)
        manifest["by_doc_language"][split] = dict(sorted(counts.items()))
        print(f"{split}: {len(dataset)} {counts} → {output_dir / split}")

    if manifest["splits"]["train"] < min_train:
        raise RuntimeError(
            f"Train bilingue tem só {manifest['splits']['train']} (mínimo {min_train}). "
            "Preencha documentation_pt e documentation_en na curadoria."
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
