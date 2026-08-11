"""Atalho para QLoRA em CUDA. No Mac use train_lora.py + configs/train_local.yaml."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "train_lora.py"
    argv = sys.argv[1:]
    if not any(arg == "--config" or arg.startswith("--config=") for arg in argv):
        argv = ["--config", "configs/default.yaml", *argv]
    if not any(arg == "--backend" or arg.startswith("--backend=") for arg in argv):
        argv = [*argv, "--backend", "cuda"]
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
