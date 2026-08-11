from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from legacy_doc.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA/QLoRA com monitoramento (CUDA, MPS ou CPU)."
    )
    parser.add_argument("--config", default="configs/train_local.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Dispositivo de treino. auto escolhe cuda > mps > cpu.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma do último checkpoint em output_dir, se existir.",
    )
    return parser.parse_args()


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA solicitada, mas não está disponível.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS solicitado, mas não está disponível.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _progress_bar(percent: float, width: int = 28) -> str:
    filled = int(width * min(max(percent, 0.0), 100.0) / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class MetricsMonitorCallback(TrainerCallback):
    """Grava CSV + JSON + STATUS.txt ao vivo para acompanhar o treino."""

    def __init__(self, csv_path: Path, live_path: Path, status_path: Path) -> None:
        self.csv_path = csv_path
        self.live_path = live_path
        self.status_path = status_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "step",
                        "epoch",
                        "loss",
                        "eval_loss",
                        "learning_rate",
                        "elapsed_seconds",
                    ],
                )
                writer.writeheader()

    def _write_status(self, text: str) -> None:
        self.status_path.write_text(text + "\n", encoding="utf-8")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        msg = (
            f"TREINO INICIADO\n"
            f"steps planejados: {state.max_steps}\n"
            f"acompanhe: python scripts/watch_training.py"
        )
        self._write_status(msg)
        print(msg, flush=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        elapsed = round(time.perf_counter() - self._started, 2)
        total = max(int(state.max_steps or 0), 1)
        percent = round(100 * state.global_step / total, 2)
        row = {
            "step": state.global_step,
            "epoch": round(float(logs.get("epoch", state.epoch or 0)), 4),
            "loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
            "elapsed_seconds": elapsed,
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writerow(row)
        payload = {
            **row,
            "total_steps": state.max_steps,
            "percent": percent,
            "logs": logs,
            "status": "running",
        }
        self.live_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        loss = logs.get("loss", logs.get("eval_loss"))
        status = (
            f"{_progress_bar(percent)} {percent:.1f}%\n"
            f"step {state.global_step}/{state.max_steps}\n"
            f"loss {loss}\n"
            f"elapsed {elapsed}s\n"
            f"status RUNNING"
        )
        self._write_status(status)
        print(f"[monitor] {_progress_bar(percent)} {percent:.1f}% "
              f"step={state.global_step}/{state.max_steps} loss={loss} "
              f"elapsed={elapsed}s", flush=True)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        msg = f"TREINO CONCLUÍDO\nsteps: {state.global_step}\nstatus DONE"
        self._write_status(msg)
        print(msg, flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg, train_cfg = cfg["model"], cfg["training"]
    model_name = args.model or model_cfg["name"]
    output_dir = Path(args.output_dir or cfg["project"]["output_dir"])
    set_seed(cfg["project"]["seed"])

    backend = resolve_backend(args.backend)
    use_qlora = backend == "cuda" and bool(model_cfg.get("load_in_4bit", False))
    print(f"Backend: {backend} | QLoRA: {use_qlora}")
    if backend == "cpu":
        print("Aviso: treino em CPU será muito lento.")
    if backend == "mps":
        # Evita estouro agressivo do cache MPS em Mac 8GB.
        import os

        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        print(
            "Aviso: M1/MPS — max_length baixo, eval desligado e save frequente "
            "recomendados (configs/train_real.yaml)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_qlora:
        compute_dtype = torch.bfloat16 if train_cfg.get("bf16") else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=model_cfg["trust_remote_code"],
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"]
        )
        bf16 = bool(train_cfg.get("bf16"))
        fp16 = bool(train_cfg.get("fp16"))
    else:
        # MPS/CPU: LoRA sem quantização 4-bit. float16 no MPS; float32 no CPU.
        torch_dtype = torch.float16 if backend == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=model_cfg["trust_remote_code"],
        )
        if backend == "mps":
            model = model.to("mps")
        if train_cfg["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        bf16 = False
        fp16 = False

    model.config.use_cache = False
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
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")
    min_train = int(train_cfg.get("min_train_samples", 0))
    if min_train and len(train_dataset) < min_train:
        raise RuntimeError(
            f"Dataset pequeno demais para treino real: {len(train_dataset)} < {min_train}. "
            "Use configs/train_real.yaml + scripts/build_sft_from_codexglue.py "
            "(não o smoke)."
        )

    logging_dir = Path(train_cfg.get("logging_dir", "outputs/training/runs"))
    metrics_csv = Path(train_cfg.get("metrics_csv", "outputs/training/metrics.csv"))
    monitor_dir = metrics_csv.parent
    live_json = monitor_dir / "live.json"
    status_txt = monitor_dir / "STATUS.txt"
    report_to = train_cfg.get("report_to", ["tensorboard"])
    if isinstance(report_to, str):
        report_to = [report_to]

    print("=" * 60, flush=True)
    print("Iniciando treino — para acompanhar em outro Terminal:", flush=True)
    print("  python scripts/watch_training.py", flush=True)
    print("  (ou abra outputs/training/STATUS.txt)", flush=True)
    print("=" * 60, flush=True)
    status_txt.parent.mkdir(parents=True, exist_ok=True)
    status_txt.write_text(
        "PREPARANDO (download do modelo / carregar dataset)...\n"
        "Se esta mensagem mudar, o treino está avançando.\n",
        encoding="utf-8",
    )

    eval_strategy = train_cfg.get("eval_strategy", "steps")
    if backend == "mps" and eval_strategy == "steps":
        # Eval em lote no MPS costuma estourar memória unificada de 8GB.
        print("MPS: forçando eval_strategy=no para reduzir OOM.")
        eval_strategy = "no"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        max_length=model_cfg["max_length"],
        dataset_text_field="text",
        report_to=report_to,
        logging_dir=str(logging_dir),
        logging_first_step=True,
        seed=cfg["project"]["seed"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    if backend == "cpu":
        try:
            training_args.use_cpu = True
        except Exception:
            pass

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricsMonitorCallback(metrics_csv, live_json, status_txt)],
    )

    resume = args.resume
    if resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        resume_path = str(checkpoints[-1]) if checkpoints else None
        if resume_path:
            print(f"Retomando de {resume_path}", flush=True)
        else:
            print("Nenhum checkpoint para retomar; treino do zero.", flush=True)
            resume_path = None
    else:
        resume_path = None

    try:
        result = trainer.train(resume_from_checkpoint=resume_path)
    except Exception as exc:
        status_txt.write_text(f"TREINO FALHOU\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        err_path = monitor_dir / "error.txt"
        err_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"Erro gravado em {err_path}", flush=True)
        raise

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {**result.metrics, "backend": backend, "qlora": use_qlora}
    if eval_strategy != "no":
        metrics.update(trainer.evaluate())
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Adapter salvo em {output_dir}")
    print(f"Métricas CSV: {metrics_csv}")
    print(f"TensorBoard: tensorboard --logdir {logging_dir}")


if __name__ == "__main__":
    main()
