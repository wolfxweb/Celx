from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from legacy_doc.prompts import SYSTEM_PROMPT, user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o mesmo benchmark nos modelos candidatos.")
    parser.add_argument("--config", default="configs/model_candidates.yaml")
    parser.add_argument("--model", action="append", help="ID ou short_name; pode ser repetido")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def apply_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def multimodal_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


def select_candidates(config: dict[str, Any], selected: list[str] | None) -> list[dict[str, Any]]:
    candidates = [item for item in config["candidates"] if item.get("enabled", True)]
    if not selected:
        return candidates
    requested = set(selected)
    matches = [
        item
        for item in candidates
        if item["id"] in requested or item["short_name"] in requested
    ]
    found = {value for item in matches for value in (item["id"], item["short_name"])}
    missing = requested - found
    if missing:
        raise ValueError(f"Candidatos não encontrados: {', '.join(sorted(missing))}")
    return matches


def main() -> None:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    benchmark_cfg = config["benchmark"]
    examples = read_jsonl(Path(benchmark_cfg["dataset"]))
    if args.limit:
        examples = examples[: args.limit]
    candidates = select_candidates(config, args.model)
    output_dir = Path(benchmark_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        model_id = candidate["id"]
        architecture = AutoConfig.from_pretrained(model_id).model_type
        is_mistral3 = architecture == "mistral3"
        processor = AutoProcessor.from_pretrained(model_id) if is_mistral3 else None
        tokenizer = None if is_mistral3 else AutoTokenizer.from_pretrained(model_id)
        if tokenizer is not None:
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        quantization = None
        if torch.cuda.is_available() and not args.no_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model_class = AutoModelForImageTextToText if is_mistral3 else AutoModelForCausalLM
        model = model_class.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        destination = output_dir / f"{candidate['short_name']}.jsonl"
        completed_ids: set[str] = set()
        if destination.exists() and not args.overwrite:
            completed_ids = {record["id"] for record in read_jsonl(destination)}
        processed = len(completed_ids)
        print(f"{candidate['short_name']}: retomando em {processed}/{len(examples)}")
        mode = "w" if args.overwrite else "a"
        with destination.open(mode, encoding="utf-8") as stream:
            for example in examples:
                if example["id"] in completed_ids:
                    continue
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt(example["language"], example["code"]),
                    },
                ]
                if is_mistral3:
                    inputs = processor.apply_chat_template(
                        multimodal_messages(messages),
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    ).to(model.device)
                else:
                    prompt = apply_template(tokenizer, messages)
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=benchmark_cfg["max_new_tokens"],
                        do_sample=False,
                        pad_token_id=(
                            processor.tokenizer.eos_token_id
                            if is_mistral3
                            else tokenizer.eos_token_id
                        ),
                    )
                elapsed = time.perf_counter() - started
                new_tokens = generated[0, inputs["input_ids"].shape[1] :]
                decoder = processor if is_mistral3 else tokenizer
                response = decoder.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[
                    0
                ].strip()
                record = {
                    **example,
                    "model": model_id,
                    "response": response,
                    "elapsed_seconds": round(elapsed, 3),
                    "generated_tokens": int(new_tokens.shape[0]),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                processed += 1
                if processed % 10 == 0 or processed == len(examples):
                    print(f"{candidate['short_name']}: {processed}/{len(examples)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Resultado salvo em {destination}")


if __name__ == "__main__":
    main()
