from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy_doc.config import load_config
from legacy_doc.prompts import system_prompt, user_prompt
from legacy_doc.thinking import apply_chat_template, extract_thinking, strip_thinking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera documentação para uma função legada.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--code")
    parser.add_argument(
        "--language", required=True, choices=["python", "php", "javascript", "sql"]
    )
    parser.add_argument(
        "--doc-language",
        default=None,
        help="Idioma da documentação: pt-BR ou en (default: generation.doc_language ou pt-BR).",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Imprime também o raciocínio bruto do Qwen3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg["model"]["name"]
    code = args.code if args.code is not None else args.file.read_text(encoding="utf-8")
    doc_language = args.doc_language or cfg.get("generation", {}).get("doc_language", "pt-BR")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = None
    else:
        dtype = torch.float32
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if device_map is None and torch.backends.mps.is_available():
        model = model.to("mps")
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))

    messages = [
        {"role": "system", "content": system_prompt(doc_language)},
        {"role": "user", "content": user_prompt(args.language, code, doc_language)},
    ]
    enable_thinking = bool(cfg.get("generation", {}).get("enable_thinking", True))
    prompt = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
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
    raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    documentation = strip_thinking(raw)
    if args.show_thinking:
        thinking = extract_thinking(raw)
        if thinking:
            print("=== Raciocínio ===")
            print(thinking)
            print("=== Documentação ===")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(documentation.strip() + "\n", encoding="utf-8")
    print(documentation.strip())


if __name__ == "__main__":
    main()
