from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from legacy_doc.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA para documentação de legado.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg, train_cfg = cfg["model"], cfg["training"]
    model_name = args.model or model_cfg["name"]
    output_dir = args.output_dir or cfg["project"]["output_dir"]
    set_seed(cfg["project"]["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("O treinamento QLoRA exige uma GPU CUDA. Use o Google Colab com GPU.")

    compute_dtype = torch.bfloat16 if train_cfg["bf16"] else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"]
    )

    peft_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=train_cfg["target_modules"],
    )
    data_dir = Path(cfg["data"]["sft_dir"])
    train_dataset = load_from_disk(str(data_dir / "train"))
    eval_dataset = load_from_disk(str(data_dir / "validation"))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        max_length=model_cfg["max_length"],
        dataset_text_field="text",
        report_to="none",
        seed=cfg["project"]["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {**result.metrics, **trainer.evaluate()}
    Path(output_dir, "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Adapter salvo em {output_dir}")


if __name__ == "__main__":
    main()
