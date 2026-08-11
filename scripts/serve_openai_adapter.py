from __future__ import annotations

"""
Servidor OpenAI-compatible local para o adapter Celx (VS Code Continue / clientes locais).

Uso:
  source .venv/bin/activate
  python scripts/serve_openai_adapter.py

API: http://127.0.0.1:8000/v1
Modelo id: celx-legacy-doc

No Cursor isto NÃO resolve localhost (bloqueio SSRF). Use VS Code + Continue,
ou o app Ollama. No Cursor, desligue Override Base URL.
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy_doc.thinking import apply_chat_template, strip_thinking

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "models" / "export" / "qwen2.5-1.5b-celx" / "adapter"
MODEL_ID = "celx-legacy-doc"

_tokenizer = None
_model = None
_device = None


def load_stack(adapter: Path, base_model: str | None) -> None:
    global _tokenizer, _model, _device
    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Adapter inválido: {adapter}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = base_model or cfg.get("base_model_name_or_path") or "Qwen/Qwen3-1.7B"

    print(f"Carregando base {base} + adapter {adapter} ...")
    _tokenizer = AutoTokenizer.from_pretrained(str(adapter) if (adapter / "tokenizer_config.json").exists() else base)
    if torch.backends.mps.is_available():
        dtype = torch.float16
        _device = "mps"
    elif torch.cuda.is_available():
        dtype = torch.float16
        _device = "cuda"
    else:
        dtype = torch.float32
        _device = "cpu"

    _model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    _model = PeftModel.from_pretrained(_model, str(adapter))
    _model.to(_device)
    _model.eval()
    print(f"Pronto em {_device}. POST /v1/chat/completions  model={MODEL_ID}")


def chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens") or body.get("max_new_tokens") or 512)
    temperature = float(body.get("temperature") or 0.2)
    enable_thinking = True

    # Continue / clientes OpenAI mandam roles system/user/assistant
    prompt = apply_chat_template(_tokenizer, messages, enable_thinking=enable_thinking)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )
    raw = _tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    text = strip_thinking(raw).strip()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/v1/models", "/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "owned_by": "celx",
                        }
                    ],
                },
            )
            return
        if path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": MODEL_ID})
            return
        self._json(404, {"error": {"message": f"not found: {path}", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid json", "type": "invalid_request"}})
            return

        if path in ("/v1/chat/completions", "/chat/completions"):
            if body.get("stream"):
                # Continue às vezes pede stream; respondemos em um único chunk SSE.
                try:
                    result = chat_completion(body)
                except Exception as exc:  # noqa: BLE001
                    self._json(500, {"error": {"message": str(exc), "type": "server_error"}})
                    return
                content = result["choices"][0]["message"]["content"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk = {
                    "id": result["id"],
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                done = {
                    "id": result["id"],
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            try:
                self._json(200, chat_completion(body))
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": {"message": str(exc), "type": "server_error"}})
            return

        self._json(404, {"error": {"message": f"not found: {path}", "type": "not_found"}})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="API OpenAI local para o adapter Celx.")
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--base-model", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_stack(args.adapter.resolve(), args.base_model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Servindo http://{args.host}:{args.port}/v1")
    print("VS Code Continue: provider openai, apiBase esta URL, model celx-legacy-doc")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main()
