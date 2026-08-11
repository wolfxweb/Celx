# Funções do pipeline (inline) — adaptadas da lógica do repositório.
# Usa CFG (dict) embutido no setup; prompts/thinking já carregados na célula anterior.
# Nenhuma etapa chama scripts externos via subprocess para o pipeline.

from __future__ import annotations

import csv
import hashlib
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# prompts + thinking já definidos na célula anterior (inline).
# ---------------------------------------------------------------------------
# A1 — SQL (Spider)
# ---------------------------------------------------------------------------


def _sql_fingerprint(code: str) -> str:
    return hashlib.sha256(f"sql\0{code.strip()}".encode()).hexdigest()


def _sql_to_row(query: str, question: str) -> dict[str, Any] | None:
    code = str(query).strip()
    reference = str(question).strip()
    if len(code) < 8 or len(reference) < 8:
        return None
    return {
        "id": _sql_fingerprint(code),
        "language": "sql",
        "code": code,
        "reference": reference,
        "source": "xlangai/spider",
        "license": "CC-BY-SA-4.0",
    }


def prepare_sql_spider(cfg: dict, root: Path | None = None) -> dict[str, Any]:
    """Baixa Spider e gera dataset/sql/{train,validation,test}.jsonl."""
    root = root or Path.cwd()
    cfg = cfg
    data_cfg = cfg["data"]
    seed = int(cfg["project"]["seed"])
    random.seed(seed)

    out = Path(data_cfg.get("sql_dataset_path", "dataset/sql"))
    if not out.is_absolute():
        out = root / out
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
        item = _sql_to_row(row["query"], row["question"])
        if item is None or item["id"] in seen:
            continue
        seen.add(item["id"])
        candidates.append(item)
        if len(candidates) >= needed + 50:
            break

    if len(candidates) < needed:
        n = len(candidates)
        print(
            f"Aviso: Spider rendeu {n} únicos (pedido {needed}). "
            "Distribuindo o disponível em test/val/train."
        )
        n_test = min(limits["test"], max(1, n // 10))
        n_val = min(limits["validation"], max(1, n // 10))
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

    offset = 0
    manifest: dict[str, Any] = {"source": "xlangai/spider", "seed": seed, "splits": {}}
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
    return manifest


# ---------------------------------------------------------------------------
# A2 — CodeXGLUE + merge SQL
# ---------------------------------------------------------------------------


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            if isinstance(value, list):
                return " ".join(str(item) for item in value)
            return str(value)
    return ""


def _normalize_codexglue(row: dict[str, Any], language: str) -> dict[str, Any]:
    code = _first_present(row, ("code", "original_string", "function"))
    reference = _first_present(row, ("docstring", "documentation", "summary"))
    fingerprint = hashlib.sha256(f"{language}\0{code}".encode()).hexdigest()
    return {
        "id": fingerprint,
        "language": language,
        "code": code.strip(),
        "reference": reference.strip(),
    }


def _valid_code_ref(row: dict[str, Any], minimum: int, maximum: int) -> bool:
    return minimum <= len(row["code"]) <= maximum and len(row["reference"]) >= 8


def _prepare_codexglue_split(
    dataset_name: str,
    split: str,
    languages: list[str],
    limit: int,
    data_cfg: dict[str, Any],
    seed: int,
    excluded_ids: set[str],
) -> Dataset:
    portions: list[Dataset] = []
    for language in languages:
        source = load_dataset(dataset_name, language, split=split)
        source = source.map(
            lambda row, lang=language: _normalize_codexglue(row, lang),
            remove_columns=source.column_names,
        )
        source = source.filter(
            lambda row: _valid_code_ref(
                row, data_cfg["min_code_chars"], data_cfg["max_code_chars"]
            )
        )
        source = source.shuffle(seed=seed).select(range(min(limit, len(source))))
        portions.append(source)
    merged = concatenate_datasets(portions).shuffle(seed=seed)
    selected: list[int] = []
    local_ids: set[str] = set()
    for index, fingerprint in enumerate(merged["id"]):
        if fingerprint not in excluded_ids and fingerprint not in local_ids:
            selected.append(index)
            local_ids.add(fingerprint)
    excluded_ids.update(local_ids)
    return merged.select(selected)


def _load_sql_split(sql_dir: Path, split: str) -> Dataset | None:
    path = sql_dir / f"{split}.jsonl"
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "id": row["id"],
                    "language": "sql",
                    "code": str(row["code"]).strip(),
                    "reference": str(row["reference"]).strip(),
                }
            )
    return Dataset.from_list(rows) if rows else None


def prepare_codexglue_dataset(cfg: dict, root: Path | None = None) -> dict[str, Any]:
    """Normaliza CodeXGLUE e mescla SQL se pedido no config."""
    root = root or Path.cwd()
    cfg = cfg
    random.seed(cfg["project"]["seed"])
    model_name = cfg["model"]["name"]
    data_cfg = cfg["data"]
    output = Path(data_cfg["normalized_dir"])
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": data_cfg["dataset_name"],
        "model_for_future_tokenization": model_name,
        "status": "normalized_not_ready_for_sft",
        "languages": list(data_cfg.get("codexglue_languages", [])),
        "sql_merged": False,
    }
    seen_ids: set[str] = set()
    prepared_splits: dict[str, Dataset] = {}
    for split in ("test", "validation", "train"):
        limit = data_cfg[f"max_{split}_samples_per_language"]
        prepared_splits[split] = _prepare_codexglue_split(
            data_cfg["dataset_name"],
            split,
            data_cfg["codexglue_languages"],
            limit,
            data_cfg,
            cfg["project"]["seed"],
            seen_ids,
        )

    sql_dir = Path(data_cfg.get("sql_dataset_path", "dataset/sql"))
    if not sql_dir.is_absolute():
        sql_dir = root / sql_dir
    want_sql = "sql" in [str(x).lower() for x in data_cfg.get("target_languages", [])]
    if want_sql:
        sql_parts: list[str] = []
        for split in ("train", "validation", "test"):
            sql_ds = _load_sql_split(sql_dir, split)
            if sql_ds is None:
                continue
            keep = [i for i, item_id in enumerate(sql_ds["id"]) if item_id not in seen_ids]
            if not keep:
                continue
            sql_ds = sql_ds.select(keep)
            seen_ids.update(sql_ds["id"])
            prepared_splits[split] = concatenate_datasets(
                [prepared_splits[split], sql_ds]
            ).shuffle(seed=cfg["project"]["seed"])
            sql_parts.append(f"{split}:{len(sql_ds)}")
        if sql_parts:
            manifest["sql_merged"] = True
            manifest["sql_source"] = str(sql_dir)
            manifest["sql_counts"] = sql_parts
            manifest["languages"] = list(dict.fromkeys([*manifest["languages"], "sql"]))
        else:
            print(
                f"Aviso: SQL pedido em target_languages, mas sem JSONL em {sql_dir}. "
                "Rode prepare_sql_spider() antes."
            )

    for split in ("train", "validation", "test"):
        prepared = prepared_splits[split]
        prepared.save_to_disk(str(output / split))
        prepared.to_json(output / f"{split}.jsonl", force_ascii=False)
        languages = sorted(set(prepared["language"]))
        manifest[split] = {
            "rows": len(prepared),
            "path": str(output / split),
            "languages": languages,
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# A3 — SFT a partir do normalizado
# ---------------------------------------------------------------------------


def _render_sft_row(
    row: dict[str, Any], tokenizer: AutoTokenizer, doc_language: str
) -> dict[str, Any]:
    messages = training_messages(
        row["language"], row["code"], row["reference"], doc_language=doc_language
    )
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return {
        "id": row["id"],
        "language": row["language"],
        "doc_language": doc_language,
        "code": row["code"],
        "reference": row["reference"],
        "messages": messages,
        "text": text,
        "source": "code_x_glue",
    }


def build_sft_from_normalized(
    cfg: dict,
    doc_language: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Constrói SFT com training_messages + doc_language."""
    root = root or Path.cwd()
    cfg = cfg
    data_cfg = cfg["data"]
    normalized = Path(data_cfg["normalized_dir"])
    output_dir = Path(data_cfg["sft_dir"])
    if not normalized.is_absolute():
        normalized = root / normalized
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    min_train = int(cfg.get("training", {}).get("min_train_samples", 100))
    doc_language = doc_language or cfg.get("generation", {}).get("doc_language", "pt-BR")

    if not (normalized / "train").exists():
        raise FileNotFoundError(
            f"Normalizado ausente em {normalized}. Rode prepare_codexglue_dataset() antes."
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model": cfg["model"]["name"],
        "source": "normalized_codexglue_plus_sql",
        "normalized_dir": str(normalized),
        "kind": "full_pipeline",
        "doc_language": doc_language,
        "splits": {},
        "languages": {},
    }
    for split in ("train", "validation", "test"):
        source = load_from_disk(str(normalized / split))
        rendered = [_render_sft_row(row, tokenizer, doc_language) for row in source]
        dataset = Dataset.from_list(rendered)
        dataset.save_to_disk(str(output_dir / split))
        dataset.to_json(output_dir / f"{split}.jsonl", force_ascii=False)
        langs: dict[str, int] = {}
        for lang in dataset["language"]:
            langs[lang] = langs.get(lang, 0) + 1
        manifest["splits"][split] = len(dataset)
        manifest["languages"][split] = dict(sorted(langs.items()))
        print(f"{split}: {len(dataset)} exemplos {langs} → {output_dir / split}")

    if manifest["splits"]["train"] < min_train:
        raise RuntimeError(
            f"Train tem só {manifest['splits']['train']} exemplos (mínimo {min_train})."
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# A4 — LoRA / QLoRA + MetricsMonitorCallback
# ---------------------------------------------------------------------------


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

    def on_train_begin(self, args, state, control, **kwargs):
        msg = (
            f"TREINO INICIADO\n"
            f"steps planejados: {state.max_steps}\n"
            f"acompanhe: STATUS.txt / live.json / metrics.csv neste MONITOR"
        )
        self._write_status(msg)
        print(msg, flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
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
            csv.DictWriter(stream, fieldnames=list(row)).writerow(row)
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
        print(
            f"[monitor] {_progress_bar(percent)} {percent:.1f}% "
            f"step={state.global_step}/{state.max_steps} loss={loss} "
            f"elapsed={elapsed}s",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        msg = f"TREINO CONCLUÍDO\nsteps: {state.global_step}\nstatus DONE"
        self._write_status(msg)
        print(msg, flush=True)


def resolve_backend(requested: str = "auto") -> str:
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


def train_lora(
    cfg: dict,
    backend: str = "auto",
    output_dir: Path | None = None,
    resume: bool = True,
    root: Path | None = None,
) -> Path:
    """Fine-tuning LoRA (MPS/CPU) ou QLoRA (CUDA + load_in_4bit). Roda in-process."""
    root = root or Path.cwd()
    cfg = cfg
    model_cfg, train_cfg = cfg["model"], cfg["training"]
    model_name = model_cfg["name"]
    out = Path(output_dir or cfg["project"]["output_dir"])
    if not out.is_absolute():
        out = root / out
    set_seed(cfg["project"]["seed"])

    backend = resolve_backend(backend)
    use_qlora = backend == "cuda" and bool(model_cfg.get("load_in_4bit", False))
    print(f"Backend: {backend} | QLoRA: {use_qlora}")
    if backend == "cpu":
        print("Aviso: treino em CPU será muito lento.")
    if backend == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

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
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    train_dataset = load_from_disk(str(data_dir / "train"))
    eval_dataset = load_from_disk(str(data_dir / "validation"))
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")
    min_train = int(train_cfg.get("min_train_samples", 0))
    if min_train and len(train_dataset) < min_train:
        raise RuntimeError(f"Dataset pequeno demais: {len(train_dataset)} < {min_train}.")

    logging_dir = Path(train_cfg.get("logging_dir", "outputs/training/runs"))
    metrics_csv = Path(train_cfg.get("metrics_csv", "outputs/training/metrics.csv"))
    if not logging_dir.is_absolute():
        logging_dir = root / logging_dir
    if not metrics_csv.is_absolute():
        metrics_csv = root / metrics_csv
    monitor_dir = metrics_csv.parent
    live_json = monitor_dir / "live.json"
    status_txt = monitor_dir / "STATUS.txt"
    report_to = train_cfg.get("report_to", ["tensorboard"])
    if isinstance(report_to, str):
        report_to = [report_to]

    print("=" * 60, flush=True)
    print(f"MONITOR: {monitor_dir}", flush=True)
    print("Arquivos: STATUS.txt | live.json | metrics.csv", flush=True)
    print("=" * 60, flush=True)
    status_txt.parent.mkdir(parents=True, exist_ok=True)
    status_txt.write_text(
        "PREPARANDO (download do modelo / carregar dataset)...\n"
        "Se esta mensagem mudar, o treino está avançando.\n",
        encoding="utf-8",
    )

    eval_strategy = train_cfg.get("eval_strategy", "steps")
    if backend == "mps" and eval_strategy == "steps":
        print("MPS: forçando eval_strategy=no para reduzir OOM.")
        eval_strategy = "no"

    training_args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=int(train_cfg.get("eval_steps", train_cfg.get("save_steps", 500))),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_total_limit=int(train_cfg.get("save_total_limit", 1)),
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

    resume_path = None
    if resume:
        checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            resume_path = str(checkpoints[-1])
            print(f"Retomando de {resume_path}", flush=True)
        else:
            print("Nenhum checkpoint para retomar; treino do zero.", flush=True)

    try:
        result = trainer.train(resume_from_checkpoint=resume_path)
    except Exception as exc:
        status_txt.write_text(
            f"TREINO FALHOU\n{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        err_path = monitor_dir / "error.txt"
        err_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"Erro gravado em {err_path}", flush=True)
        raise

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metrics = {**result.metrics, "backend": backend, "qlora": use_qlora}
    if eval_strategy != "no":
        metrics.update(trainer.evaluate())
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if status_txt.exists():
        print("--- STATUS.txt ---")
        print(status_txt.read_text(encoding="utf-8"))
    print(f"Adapter salvo em {out}")
    return out


# ---------------------------------------------------------------------------
# A5 — Avaliação
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = (
    "### Objetivo",
    "### Parâmetros",
    "### Retorno",
    "### Funcionamento",
    "### Regras de negócio identificadas",
    "### Pontos não determinados",
)


def _section_score(text: str) -> dict[str, Any]:
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


def _pick_eval_samples(dataset, limit: int) -> list[dict[str, Any]]:
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        by_lang[row["language"]].append(dict(row))
    languages = sorted(by_lang)
    if not languages:
        return []
    per = max(1, limit // len(languages))
    selected: list[dict[str, Any]] = []
    for lang in languages:
        rows = sorted(by_lang[lang], key=lambda r: r["id"])
        step = max(1, len(rows) // per) if len(rows) > per else 1
        selected.extend(rows[::step][:per])
    return selected[:limit]


def evaluate_adapter(
    cfg: dict,
    adapter: Path,
    output: Path | None = None,
    limit: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Avalia adapter LoRA em amostras do split test do SFT."""
    root = root or Path.cwd()
    cfg = cfg
    eval_cfg = cfg.get("evaluation", {})
    limit = limit or int(eval_cfg.get("max_samples", 12))
    out_dir = output or Path(eval_cfg.get("output_dir", "outputs/eval_full"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_test = Path(cfg["data"]["sft_dir"]) / "test"
    if not sft_test.is_absolute():
        sft_test = root / sft_test
    if not sft_test.exists():
        raise FileNotFoundError(f"Test SFT ausente: {sft_test}")

    adapter = Path(adapter).resolve()
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter inválido: {adapter}")

    test_ds = load_from_disk(str(sft_test))
    samples = _pick_eval_samples(test_ds, limit)
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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        doc = strip_thinking(raw).strip()
        score = _section_score(doc)
        results.append(
            {
                "id": row["id"],
                "language": row["language"],
                "reference": row.get("reference", ""),
                "prediction": doc,
                **score,
            }
        )
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
            lang: {"n": len(vals), "mean_section_ratio": sum(vals) / len(vals)}
            for lang, vals in sorted(by_lang.items())
        },
        "adapter": str(adapter),
        "predictions": str(results_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# A6 — Export adapter
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen2.5:1.5b-celx"
OLLAMA_BASE = "qwen2.5:1.5b"


def _write_modelfile(out: Path, ollama_base: str) -> Path:
    path = out / "Modelfile"
    system = SYSTEM_PROMPT.replace('"""', "'''")
    content = "FROM " + ollama_base + "\n\nSYSTEM \"\"\"" + system + "\"\"\"\n"
    path.write_text(content, encoding="utf-8")
    return path


def export_adapter(
    adapter: Path,
    output: Path,
    *,
    skip_ollama: bool = True,
    ollama_model: str = OLLAMA_MODEL,
    merge: bool = False,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Copia adapter + Modelfile + README + export_meta.json. skip_ollama=True por padrão."""
    adapter = Path(adapter).resolve()
    config_path = adapter / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter não encontrado em {adapter}.")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base = base_model or cfg.get("base_model_name_or_path") or "Qwen/Qwen3-1.7B"
    out = Path(output).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    adapter_out = out / "adapter"
    adapter_out.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        src = adapter / name
        if src.exists():
            shutil.copy2(src, adapter_out / name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        src = adapter / name
        if src.exists():
            shutil.copy2(src, adapter_out / name)

    modelfile = _write_modelfile(out, OLLAMA_BASE)
    ollama_ok = False
    if not skip_ollama:
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            print(f"Registrando no Ollama: {ollama_model}")
            result = subprocess.run(
                [ollama_bin, "create", ollama_model, "-f", str(modelfile)],
                check=False,
            )
            ollama_ok = result.returncode == 0
        else:
            print("Ollama não encontrado — Modelfile salvo.")

    meta = {
        "base_model": base,
        "adapter_dir": "adapter",
        "merged": False,
        "ollama_model": ollama_model,
        "ollama_base": OLLAMA_BASE,
        "ollama_registered": ollama_ok,
        "how_to_use": f"document_code(..., adapter={adapter_out})",
        "how_to_use_ollama": f"ollama run {ollama_model}",
    }

    if merge:
        print(f"Mesclando {base} + {adapter}...")
        tok = AutoTokenizer.from_pretrained(str(adapter))
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype="auto", low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(model, str(adapter))
        merged = model.merge_and_unload()
        merged_dir = out / "merged"
        merged_dir.mkdir()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tok.save_pretrained(str(merged_dir))
        meta["merged"] = True
        meta["merged_dir"] = "merged"

    usage = (
        f"# Modelo exportado — Celx\n\n"
        f"Base HF: `{base}`\n"
        f"Adapter: `adapter/`\n"
        f"Ollama: `{ollama_model}` (FROM `{OLLAMA_BASE}` + system Celx)\n\n"
        f"## Ollama / Continue\n\n"
        f"```bash\n"
        f"ollama create {ollama_model} -f {modelfile}\n"
        f"ollama run {ollama_model}\n"
        f"```\n\n"
        f"## Uso com adapter (neste notebook)\n\n"
        f'Chame `document_code(file=..., language=..., adapter=Path("{adapter_out}"))`.\n'
    )
    (out / "README.md").write_text(usage, encoding="utf-8")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Export pronto: {out}")
    return meta


# ---------------------------------------------------------------------------
# Stage B — SFT bilingue (curated_pairs only)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _fp(language: str, code: str) -> str:
    return hashlib.sha256(f"{language}\0{code.strip()}".encode()).hexdigest()


def _apply_template(messages: list[dict[str, str]], tokenizer: AutoTokenizer) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _render_bilingual_pair(
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
        "text": _apply_template(messages, tokenizer),
        "source": source,
    }


def build_sft_bilingual_curated(
    cfg: dict,
    root: Path | None = None,
) -> dict[str, Any]:
    """SFT bilingue simplificado: só curated_pairs (pt-BR + en quando existirem)."""
    root = root or Path.cwd()
    cfg = cfg
    data_cfg = cfg["data"]
    bilingual = cfg.get("bilingual") or {}
    doc_languages = [
        normalize_doc_language(x)
        for x in bilingual.get("doc_languages", ["pt-BR", "en"])
    ]
    curated_dir = Path(data_cfg["curated_dir"])
    output_dir = Path(data_cfg["sft_dir"])
    if not curated_dir.is_absolute():
        curated_dir = root / curated_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    min_train = int(cfg.get("training", {}).get("min_train_samples", 1))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    output_dir.mkdir(parents=True, exist_ok=True)

    curated_map: dict[str, list[dict[str, Any]]] = {
        s: [] for s in ("train", "validation", "test")
    }
    for split in curated_map:
        path = curated_dir / f"{split}.jsonl"
        for line, row in enumerate(_read_jsonl(path), start=1):
            if row.get("review_status") and row.get("review_status") != "approved":
                continue
            language = str(row["language"])
            code = str(row["code"])
            base_id = str(row.get("id") or _fp(language, code))
            for doc_lang in doc_languages:
                field = documentation_field_for(doc_lang)
                doc = str(row.get(field, "")).strip()
                if (
                    not doc
                    and doc_lang == "en"
                    and str(row.get("source_reference", "")).strip()
                ):
                    messages = training_messages(
                        language, code, str(row["source_reference"]), doc_language="en"
                    )
                    doc = messages[-1]["content"]
                    source = "curated_en_from_reference"
                elif not doc:
                    continue
                else:
                    source = "curated_bilingual"
                    missing = [s for s in required_sections(doc_lang) if s not in doc]
                    if missing:
                        raise ValueError(
                            f"{split}:{line}:{field}: seções ausentes ({doc_lang}): "
                            + ", ".join(missing)
                        )
                curated_map[split].append(
                    _render_bilingual_pair(
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

    manifest: dict[str, Any] = {
        "model": cfg["model"]["name"],
        "kind": "bilingual_sft",
        "mode": "curated_pairs",
        "doc_languages": doc_languages,
        "splits": {},
        "by_doc_language": {},
    }
    for split in ("train", "validation", "test"):
        rows = curated_map[split]
        seen: set[str] = set()
        unique = []
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
    return manifest


# ---------------------------------------------------------------------------
# Stage C — Inferência document_code
# ---------------------------------------------------------------------------


def document_code(
    *,
    file: Path | None = None,
    code: str | None = None,
    language: str,
    doc_language: str = "pt-BR",
    adapter: Path | None = None,
    cfg: dict | None = None,
    model_name: str | None = None,
    show_thinking: bool = False,
) -> str:
    """Gera documentação para um trecho de código (inline)."""
    if code is None and file is None:
        raise ValueError("Informe file= ou code=")
    cfg = cfg or globals().get("CFG")
    if cfg:
        model_name = model_name or cfg["model"]["name"]
        gen_cfg = cfg["generation"]
        enable_thinking = bool(gen_cfg.get("enable_thinking", True))
        max_new = gen_cfg["max_new_tokens"]
        temperature = gen_cfg["temperature"]
        top_p = gen_cfg["top_p"]
    else:
        model_name = model_name or "Qwen/Qwen3-1.7B"
        enable_thinking = True
        max_new = 1024
        temperature = 0.0
        top_p = 0.9

    text = code if code is not None else Path(file).read_text(encoding="utf-8")
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
        model_name, torch_dtype=dtype, device_map=device_map
    )
    if device_map is None and torch.backends.mps.is_available():
        model = model.to("mps")
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))

    messages = [
        {"role": "system", "content": system_prompt(doc_language)},
        {"role": "user", "content": user_prompt(language, text, doc_language)},
    ]
    prompt = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    documentation = strip_thinking(raw)
    if show_thinking:
        thinking = extract_thinking(raw)
        if thinking:
            print("=== Raciocínio ===")
            print(thinking)
            print("=== Documentação ===")
    print(documentation.strip())
    return documentation.strip()



# ---------------------------------------------------------------------------
# Stage B — semente curada bilingue (se pasta vazia)
# ---------------------------------------------------------------------------


def ensure_curated_bilingual_seed(root: Path | None = None, force: bool = False) -> Path:
    """Se curated estiver vazio, grava 2 exemplos bilingues inline (seed)."""
    root = root or Path.cwd()
    curated = root / "dataset" / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    train_path = curated / "train.jsonl"
    existing = 0
    for split in ("train", "validation", "test"):
        p = curated / f"{split}.jsonl"
        if p.exists():
            existing += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    if existing and not force:
        print(f"Curadoria já tem {existing} linhas — seed não sobrescrita.")
        return curated

    php_code = (
        "<?php\n\n"
        "function calculaTotal(Pedido $pedido): float\n"
        "{\n"
        "    $total = 0.0;\n"
        "    foreach ($pedido->getItens() as $item) {\n"
        "        $total += $item->getValor();\n"
        "    }\n"
        "    return $total;\n"
        "}\n"
    )
    js_code = (
        "export function calculaTotal(pedido) {\n"
        "  let total = 0;\n"
        "  for (const item of pedido.itens) {\n"
        "    total += item.valor;\n"
        "  }\n"
        "  return total;\n"
        "}\n"
    )
    doc_pt_php = """## Método ou função

Soma o valor dos itens de um pedido.

### Objetivo

Calcular o total monetário a partir dos itens do pedido.

### Parâmetros

- `$pedido`: objeto com método `getItens()` que devolve a coleção de itens.

### Retorno

`float` com a soma de `getValor()` de cada item.

### Funcionamento

- Inicializa `$total` em 0.0.
- Percorre os itens e acumula `getValor()`.
- Retorna o acumulado.

### Regras de negócio identificadas

- O total é a soma simples dos valores dos itens.

### Pontos não determinados

- Moeda, descontos, impostos e validações de item vazio não aparecem no trecho."""
    doc_en_php = """## Method or function

Sums the value of the items in an order.

### Objective

Compute the monetary total from the order items.

### Parameters

- `$pedido`: object with `getItens()` returning the item collection.

### Returns

`float` equal to the sum of each item's `getValor()`.

### Behavior

- Initializes `$total` to 0.0.
- Iterates items and accumulates `getValor()`.
- Returns the accumulated total.

### Identified business rules

- The total is a plain sum of item values.

### Undetermined points

- Currency, discounts, taxes, and empty-item validation are not shown."""
    doc_pt_js = """## Método ou função

Soma os valores dos itens de um pedido em JavaScript.

### Objetivo

Calcular o total a partir de `pedido.itens`.

### Parâmetros

- `pedido`: objeto com propriedade `itens` (iterável de itens com `valor`).

### Retorno

Número com a soma dos `valor` de cada item.

### Funcionamento

- Inicializa `total` em 0.
- Percorre `pedido.itens` acumulando `item.valor`.
- Retorna o acumulado.

### Regras de negócio identificadas

- O total é a soma simples dos valores.

### Pontos não determinados

- Tipagem, descontos e validações não estão no trecho."""
    doc_en_js = """## Method or function

Sums the item values of an order in JavaScript.

### Objective

Compute the total from `pedido.itens`.

### Parameters

- `pedido`: object with `itens` (iterable of items with `valor`).

### Returns

Number equal to the sum of each item's `valor`.

### Behavior

- Initializes `total` to 0.
- Iterates `pedido.itens` accumulating `item.valor`.
- Returns the accumulated total.

### Identified business rules

- The total is a plain sum of values.

### Undetermined points

- Typing, discounts, and validation are not in the snippet."""

    seeds = {
        "train": [
            {
                "id": "seed-php-calculaTotal",
                "language": "php",
                "code": php_code,
                "source_reference": "Sum the values of the order items.",
                "documentation_pt": doc_pt_php,
                "documentation_en": doc_en_php,
                "source": "notebook_seed",
                "split": "train",
                "review_status": "approved",
            },
            {
                "id": "seed-js-calculaTotal",
                "language": "javascript",
                "code": js_code,
                "source_reference": "Sum the values of the order items.",
                "documentation_pt": doc_pt_js,
                "documentation_en": doc_en_js,
                "source": "notebook_seed",
                "split": "train",
                "review_status": "approved",
            },
        ],
        "validation": [
            {
                "id": "seed-php-calculaTotal-val",
                "language": "php",
                "code": php_code,
                "source_reference": "Sum the values of the order items.",
                "documentation_pt": doc_pt_php,
                "documentation_en": doc_en_php,
                "source": "notebook_seed",
                "split": "validation",
                "review_status": "approved",
            },
        ],
        "test": [
            {
                "id": "seed-js-calculaTotal-test",
                "language": "javascript",
                "code": js_code,
                "source_reference": "Sum the values of the order items.",
                "documentation_pt": doc_pt_js,
                "documentation_en": doc_en_js,
                "source": "notebook_seed",
                "split": "test",
                "review_status": "approved",
            },
        ],
    }
    for split, rows in seeds.items():
        path = curated / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Seed {split}: {len(rows)} → {path}")
    (curated / "README.md").write_text(
        "# Curadoria (seed do notebook)\n\n"
        "Exemplos bilingues gerados automaticamente porque a pasta estava vazia.\n"
        "Substitua por pares reais pt-BR + en quando disponíveis.\n",
        encoding="utf-8",
    )
    return curated


# ---------------------------------------------------------------------------
# Inventário (simplificado)
# ---------------------------------------------------------------------------


def _summarize_jsonl_dir(directory: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(directory),
        "exists": directory.exists(),
        "splits": {},
    }
    if not directory.exists():
        return report
    total = 0
    langs: Counter[str] = Counter()
    with_pt = 0
    with_en = 0
    for split in ("train", "validation", "test"):
        rows = _read_jsonl(directory / f"{split}.jsonl")
        split_langs = Counter(str(r.get("language", "?")) for r in rows)
        pt = sum(1 for r in rows if str(r.get("documentation_pt", "")).strip())
        en = sum(1 for r in rows if str(r.get("documentation_en", "")).strip())
        report["splits"][split] = {
            "rows": len(rows),
            "languages": dict(sorted(split_langs.items())),
            "with_documentation_pt": pt,
            "with_documentation_en": en,
        }
        total += len(rows)
        langs.update(split_langs)
        with_pt += pt
        with_en += en
    report["total_rows"] = total
    report["languages"] = dict(sorted(langs.items()))
    report["with_documentation_pt"] = with_pt
    report["with_documentation_en"] = with_en
    return report


def _count_arrow(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    ds = load_from_disk(str(path))
    langs = Counter(ds["language"]) if "language" in ds.column_names else {}
    return {"rows": len(ds), "languages": dict(sorted(langs.items())), "path": str(path)}


def inventory_datasets(root: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    """Inventário simplificado pt-BR vs EN."""
    root = root or Path.cwd()
    curated = _summarize_jsonl_dir(root / "dataset" / "curated")
    smoke = _summarize_jsonl_dir(root / "dataset" / "curated_smoke")

    processed = root / "dataset" / "processed"
    layers: dict[str, Any] = {}
    for name in ("full", "codexglue", "sft", "sft_full", "sft_bilingual"):
        layer: dict[str, Any] = {
            "path": str(processed / name),
            "exists": (processed / name).exists(),
            "splits": {},
        }
        total = 0
        for split in ("train", "validation", "test"):
            info = _count_arrow(processed / name / split)
            layer["splits"][split] = info
            if info and "rows" in info:
                total += int(info["rows"])
        layer["total_rows"] = total
        layers[name] = layer

    inventory = {
        "kind": "celx_dataset_inventory",
        "pt_br": {"curated": curated, "smoke": smoke},
        "layers": layers,
        "decision_inputs": {
            "pt_br_with_doc": curated.get("with_documentation_pt", 0),
            "en_with_doc": curated.get("with_documentation_en", 0),
            "full_train": (layers.get("full", {}).get("splits", {}).get("train") or {}).get(
                "rows", 0
            ),
        },
    }

    md_lines = [
        "# Inventário de dados Celx",
        "",
        "Gerado inline no notebook 04 (lógica 100% no notebook).",
        "",
        f"- pt-BR com documentation_pt: **{inventory['decision_inputs']['pt_br_with_doc']}**",
        f"- EN com documentation_en: **{inventory['decision_inputs']['en_with_doc']}**",
        f"- full train (normalizado): **{inventory['decision_inputs']['full_train']}**",
        "",
        "## curated",
        f"- total: {curated.get('total_rows', 0)} | langs: `{curated.get('languages', {})}`",
        "",
        "## camadas processed",
    ]
    for name, layer in layers.items():
        md_lines.append(f"- `{name}`: exists={layer['exists']} total={layer['total_rows']}")

    out = output or (root / "dataset" / "curated" / "INVENTARIO.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out)
    print(json.dumps(inventory["decision_inputs"], ensure_ascii=False, indent=2))
    return inventory


print(
    "Helpers carregados:",
    "prepare_sql_spider, prepare_codexglue_dataset, build_sft_from_normalized,",
    "train_lora, evaluate_adapter, export_adapter,",
    "build_sft_bilingual_curated, ensure_curated_bilingual_seed, document_code, inventory_datasets",
)
