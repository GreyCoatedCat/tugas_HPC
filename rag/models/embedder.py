"""Wrapper model embedding (Hugging Face Transformers, PyTorch)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .worker import InferenceWorker

log = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class Embedder:
    """Encoder teks -> vektor ter-normalisasi L2 (cosine similarity = dot product).

    Baseline: FP32, batch_size dari mode (baseline = 1 chunk per forward pass).
    Semua forward pass berjalan di satu thread inferensi tetap (lihat
    `worker.py`), sehingga pemakaian model sekuensial dan VRAM tidak bertambah
    mengikuti jumlah thread pemanggil.
    """

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        dtype: str = "float32",
        max_length: int = 512,
        pooling: str = "mean",
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        if device == "cpu" and dtype == "float16":
            dtype = "float32"
        self.name = name
        self.device = device
        self.dtype_name = dtype
        self.max_length = max_length
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        t = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=DTYPES[dtype]).to(device).eval()
        self.dim = int(self.model.config.hidden_size)
        self._worker = InferenceWorker("embedder")
        log.info("Embedder %s (%s, %s) dimuat dalam %.1fs", name, device, dtype, time.perf_counter() - t)

    @torch.inference_mode()
    def _forward(self, enc: dict, fp16: bool = False) -> np.ndarray:
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        # FP16 lewat autocast: bobot tetap satu salinan FP32 (VRAM sempit), matmul memakai tensor core FP16
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and self.device.startswith("cuda")):
            hidden = self.model(**enc).last_hidden_state
        hidden = hidden.float()
        if self.pooling == "cls":
            emb = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 1,
        is_query: bool = False,
        progress: Callable[[int, int], None] | None = None,
        fp16: bool = False,
        sort_by_length: bool | None = None,
        overlap: bool | None = None,
    ) -> np.ndarray:
        """Encode teks -> vektor ter-normalisasi.

        Baseline: batch_size=1, FP32, urutan asli, tanpa overlap.
        Teroptimasi (batch_size > 1): teks diurutkan menurut panjang agar padding per batch
        minimal, FP16 autocast, dan tokenisasi batch berikutnya (CPU) tumpang-tindih dengan
        forward pass batch sekarang (GPU) memakai thread produsen + pinned memory.
        """
        prefix = self.query_prefix if is_query else self.passage_prefix
        items = [prefix + t for t in texts]
        n = len(items)
        out = np.empty((n, self.dim), dtype=np.float32)
        if n == 0:
            return out
        bs = max(1, int(batch_size))
        sort = (bs > 1) if sort_by_length is None else sort_by_length
        order = sorted(range(n), key=lambda i: len(items[i]), reverse=True) if sort else list(range(n))
        batches = [order[i : i + bs] for i in range(0, n, bs)]
        use_overlap = (len(batches) > 1 and self.device.startswith("cuda")) if overlap is None else overlap

        def tokenize(b: list[int]) -> dict:
            enc = self.tokenizer(
                [items[i] for i in b], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            return {k: v.pin_memory() for k, v in enc.items()} if use_overlap else dict(enc)

        def run() -> np.ndarray:
            done = 0
            if not use_overlap:
                for b in batches:
                    out[b] = self._forward(tokenize(b), fp16)
                    done += len(b)
                    if progress:
                        progress(done, n)
                return out
            q: queue.Queue = queue.Queue(maxsize=2)

            def producer() -> None:
                try:
                    for b in batches:
                        q.put((b, tokenize(b)))
                except BaseException as exc:  # noqa: BLE001
                    q.put(exc)
                    return
                q.put(None)

            th = threading.Thread(target=producer, name="embed-tokenizer", daemon=True)
            th.start()
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                b, enc = item
                out[b] = self._forward(enc, fp16)
                done += len(b)
                if progress:
                    progress(done, n)
            th.join()
            return out

        return self._worker.run(run)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1, is_query=True)[0]
