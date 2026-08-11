from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy_doc.config import load_config
from legacy_doc.prompts import SYSTEM_PROMPT, user_prompt
from legacy_doc.thinking import apply_chat_template, strip_thinking

REQUIRED_SECTIONS = (
    "### Objetivo",
    "### Parâmetros",
    "### Retorno",
    "### Funcionamento",
    "### Regras de negócio identificadas",
    "### Pontos não determinados",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o adapter LoRA em amostras do split test do SFT."
    )
    parser.add_argument("--config", default="configs/train_full.yaml")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def section_score(text: str) -> dict[str, Any]:
    found = [s for s in REQUIRED_SECTIONS if s.lower() in text.lower()]
    heading_ok = bool(
        re.search(r"^##\s+(Método ou função|Consulta SQL)", text, flags=re.M | re.I)
    )
    return {
        "sections_found": len(found),
        "sections_total": len(REQUIRED_SECTIONS),
        "section_ratio": len(found) / len(REQUIRED_SECTIONS),
        "has_heading": heading_ok,
        "char_len": len(text.strip()),
    }


def pick_samples(dataset, limit: int, seed: int) -> list[dict[str, Any]]:
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        by_lang[row["language"]].append(dict(row))
    languages = sorted(by_lang)
    if not languages:
        return []
    per = max(1, limit // len(languages))
    selected: list[dict[str, Any]] = []
    for lang in languages:
        rows = by_lang[lang]
        # determinístico sem depender de numpy
        rows = sorted(rows, key=lambda r: r["id"])
        step = max(1, len(rows) // per) if len(rows) > per else 1
        picked = rows[::step][:per]
        selected.extend(picked)
    selected = selected[:limit]
    return selected


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("evaluation", {})
    limit = args.limit or int(eval_cfg.get("max_samples", 12))
    out_dir = args.output or Path(eval_cfg.get("output_dir", "outputs/eval_full"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_test = Path(cfg["data"]["sft_dir"]) / "test"
    if not sft_test.exists():
        raise FileNotFoundError(f"Test SFT ausente: {sft_test}")

    adapter = args.adapter.resolve()
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter inválido: {adapter}")

    test_ds = load_from_disk(str(sft_test))
    samples = pick_samples(test_ds, limit, cfg["project"]["seed"])
    print(f"Avaliando {len(samples)} exemplos do test ({sft_test})")

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
        device = None
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = None
        device = "mps"
    else:
        dtype = torch.float32
        device_map = None
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )
    if device_map is None:
        model = model.to(device)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    gen_cfg = cfg["generation"]
    max_new = int(eval_cfg.get("max_new_tokens", min(512, gen_cfg["max_new_tokens"])))
    enable_thinking = bool(gen_cfg.get("enable_thinking", True))

    results: list[dict[str, Any]] = []
    for i, row in enumerate(samples, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row["language"], row["code"])},
        ]
        prompt = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        doc = strip_thinking(raw).strip()
        score = section_score(doc)
        record = {
            "id": row["id"],
            "language": row["language"],
            "reference": row.get("reference", ""),
            "prediction": doc,
            **score,
        }
        results.append(record)
        print(
            f"[{i}/{len(samples)}] {row['language']} "
            f"sections={score['sections_found']}/{score['sections_total']} "
            f"chars={score['char_len']}"
        )

    results_path = out_dir / "predictions.jsonl"
    with results_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_lang: dict[str, list[float]] = defaultdict(list)
    for row in results:
        by_lang[row["language"]].append(row["section_ratio"])

    summary = {
        "n": len(results),
        "mean_section_ratio": (
            sum(r["section_ratio"] for r in results) / len(results) if results else 0.0
        ),
        "mean_has_heading": (
            sum(1 for r in results if r["has_heading"]) / len(results) if results else 0.0
        ),
        "mean_chars": (
            sum(r["char_len"] for r in results) / len(results) if results else 0.0
        ),
        "per_language": {
            lang: {
                "n": len(vals),
                "mean_section_ratio": sum(vals) / len(vals),
            }
            for lang, vals in sorted(by_lang.items())
        },
        "adapter": str(adapter),
        "predictions": str(results_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Resumo:", summary_path)


if __name__ == "__main__":
    main()
