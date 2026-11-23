"""Wrapper LLM dengan beberapa backend yang bisa ditukar lewat config.

Semua backend mengeluarkan event stream yang seragam:
    {"type": "token", "text": "..."}            (berulang)
    {"type": "stats", "prompt_tokens": ..., ...} (sekali, di akhir)

Backend:
- ``ollama``       : server Ollama lokal (default; model GGUF terkuantisasi).
- ``openai``       : server OpenAI-compatible (vLLM, llama.cpp server, LM Studio, ...).
- ``transformers`` : Hugging Face Transformers langsung di proses ini (PyTorch).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMBackend(ABC):
    name: str = "base"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.1))
        self.max_tokens = int(cfg.get("max_tokens", 768))

    @abstractmethod
    def stream_chat(self, messages: list[dict[str, str]], **opts: Any) -> Iterator[dict[str, Any]]:
        ...

    def health(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "ok": True}

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.base_url = cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
        self.num_ctx = cfg.get("num_ctx")
        self.think = cfg.get("think")
        self.keep_alive = cfg.get("keep_alive")
        timeout = float(cfg.get("timeout_s", 600))
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def stream_chat(self, messages, **opts):
        options = {
            "temperature": opts.get("temperature", self.temperature),
            "num_predict": opts.get("max_tokens", self.max_tokens),
        }
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True, "options": options}
        if self.think is not None:
            payload["think"] = bool(self.think)
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        try:
            with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", "replace")[:500]
                    raise LLMError(f"Ollama HTTP {resp.status_code}: {body}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise LLMError(f"Ollama: {obj['error']}")
                    content = (obj.get("message") or {}).get("content")
                    if content:
                        yield {"type": "token", "text": content}
                    if obj.get("done"):
                        ns = 1e-6
                        yield {
                            "type": "stats",
                            "prompt_tokens": obj.get("prompt_eval_count"),
                            "completion_tokens": obj.get("eval_count"),
                            "backend_prefill_ms": (obj.get("prompt_eval_duration") or 0) * ns,
                            "backend_decode_ms": (obj.get("eval_duration") or 0) * ns,
                            "backend_load_ms": (obj.get("load_duration") or 0) * ns,
                            "done_reason": obj.get("done_reason"),
                        }
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Tidak bisa terhubung ke Ollama di {self.base_url}. Jalankan `ollama serve` terlebih dahulu."
            ) from exc

    def health(self):
        try:
            r = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name") for m in r.json().get("models", [])]
            ok = self.model in names or f"{self.model}:latest" in names
            return {
                "backend": self.name,
                "model": self.model,
                "ok": ok,
                "detail": None if ok else f"model belum di-pull (tersedia: {names})",
            }
        except Exception as exc:  # noqa: BLE001
            return {"backend": self.name, "model": self.model, "ok": False, "detail": str(exc)}


class OpenAICompatBackend(LLMBackend):
    name = "openai"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.base_url = cfg.get("base_url", "http://127.0.0.1:8001").rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        headers = {}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        timeout = float(cfg.get("timeout_s", 600))
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0), headers=headers)

    def stream_chat(self, messages, **opts):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": opts.get("temperature", self.temperature),
            "max_tokens": opts.get("max_tokens", self.max_tokens),
            "stream_options": {"include_usage": True},
        }
        usage: dict[str, Any] = {}
        with self.client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", "replace")[:500]
                raise LLMError(f"OpenAI-compatible HTTP {resp.status_code}: {body}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    content = (ch.get("delta") or {}).get("content")
                    if content:
                        yield {"type": "token", "text": content}
        yield {
            "type": "stats",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    def health(self):
        try:
            r = self.client.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            return {"backend": self.name, "model": self.model, "ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"backend": self.name, "model": self.model, "ok": False, "detail": str(exc)}


class TransformersBackend(LLMBackend):
    """Generasi langsung dengan HF Transformers (berguna untuk eksperimen batching/CUDA streams)."""

    name = "transformers"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..config import resolve_device

        self.device = resolve_device(cfg.get("device", "auto"))
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.lm = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype).to(self.device).eval()
        self._lock = threading.Lock()

    def stream_chat(self, messages, **opts):
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False
        ).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        temperature = opts.get("temperature", self.temperature)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=opts.get("max_tokens", self.max_tokens),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
        )
        n_prompt = int(inputs["input_ids"].shape[-1])
        n_out = 0
        with self._lock:
            err: list[BaseException] = []

            def _run():
                try:
                    self.lm.generate(**gen_kwargs)
                except BaseException as exc:  # noqa: BLE001
                    err.append(exc)
                    streamer.end()

            th = threading.Thread(target=_run, daemon=True)
            th.start()
            for text in streamer:
                if text:
                    n_out += 1
                    yield {"type": "token", "text": text}
            th.join()
            if err:
                raise LLMError(f"transformers: {err[0]}")
        yield {"type": "stats", "prompt_tokens": n_prompt, "completion_tokens": None}


BACKENDS = {"ollama": OllamaBackend, "openai": OpenAICompatBackend, "transformers": TransformersBackend}


def create_llm(cfg: dict[str, Any]) -> LLMBackend:
    backend = cfg.get("backend", "ollama")
    if backend not in BACKENDS:
        raise ValueError(f"Backend LLM tidak dikenal: {backend} (pilihan: {list(BACKENDS)})")
    return BACKENDS[backend](cfg)


def warmup(llm: LLMBackend) -> float:
    """Panggil LLM sekali agar model termuat ke memori; kembalikan durasi (ms)."""
    t = time.perf_counter()
    for _ in llm.stream_chat([{"role": "user", "content": "Jawab: OK"}], max_tokens=1):
        pass
    return (time.perf_counter() - t) * 1000.0
