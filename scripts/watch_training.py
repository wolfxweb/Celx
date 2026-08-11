#!/usr/bin/env python3
"""Mostra no Terminal se o treino/benchmark está andando.

Uso (outro Terminal, com o projeto como cwd):

    python scripts/watch_training.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def bar(percent: float, width: int = 28) -> str:
    filled = int(width * min(max(percent, 0.0), 100.0) / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def render(root: Path) -> str:
    candidates = [
        root / "outputs/training_php_js_sql/live.json",
        root / "outputs/training_bilingual/live.json",
        root / "outputs/training_full/live.json",
        root / "outputs/training_real/live.json",
        root / "outputs/training/live.json",
    ]
    live_path = next((p for p in candidates if p.exists()), candidates[0])
    live = read_json(live_path)
    status_txt = live_path.with_name("STATUS.txt")
    lines = ["=== Celx — status ===", f"hora: {time.strftime('%H:%M:%S')}", ""]

    if live:
        total = max(int(live.get("total_steps") or 0), 1)
        step = int(live.get("step") or 0)
        percent = float(live.get("percent") or (100 * step / total))
        loss = live.get("loss", live.get("eval_loss"))
        elapsed = live.get("elapsed_seconds")
        lines += [
            "TREINO",
            f"  {bar(percent)} {percent:.1f}%",
            f"  step {step}/{total}",
            f"  loss {loss}",
            f"  elapsed {elapsed}s",
            f"  arquivo: {live_path.relative_to(root)}",
        ]
        age = time.time() - live_path.stat().st_mtime
        if age > 90:
            lines.append(f"  AVISO: sem atualização há {int(age)}s (pode ter travado)")
        else:
            lines.append(f"  ok: atualizado há {int(age)}s")
    elif status_txt.exists():
        lines += ["TREINO (STATUS.txt)", status_txt.read_text(encoding="utf-8")]
    else:
        lines += [
            "TREINO",
            "  ainda sem live.json — o treino não começou ou está baixando o modelo.",
            "  espere mensagens no notebook / Terminal do treino.",
        ]

    # Benchmark smoke / baseline
    baseline_dir = root / "outputs/baseline"
    bench_dir = root / "outputs/benchmark_100"
    lines.append("")
    lines.append("BENCHMARK")
    found = False
    for folder in (baseline_dir, bench_dir):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            n = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            age = time.time() - path.stat().st_mtime
            lines.append(f"  {path.name}: {n} respostas (atualizado há {int(age)}s)")
            found = True
    if not found:
        lines.append("  nenhum resultado em outputs/baseline ou outputs/benchmark_100")

    lines.append("")
    lines.append("Ctrl+C para sair do watch")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acompanhar treino/benchmark em tempo real.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        while True:
            text = render(root)
            print("\033[2J\033[H", end="")  # limpa tela
            print(text)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nwatch encerrado.")


if __name__ == "__main__":
    main()
