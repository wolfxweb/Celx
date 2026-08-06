from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy_doc.config import load_config
from legacy_doc.prompts import SYSTEM_PROMPT, user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera documentação para uma função legada.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--code")
    parser.add_argument(
        "--language", required=True, choices=["python", "php", "javascript", "sql"]
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg["model"]["name"]
    code = args.code if args.code is not None else args.file.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(args.language, code)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_cfg = cfg["generation"]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["temperature"] > 0,
            temperature=max(gen_cfg["temperature"], 1e-5),
            top_p=gen_cfg["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )
    documentation = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(documentation.strip() + "\n", encoding="utf-8")
    print(documentation.strip())


if __name__ == "__main__":
    main()
