"""Wrapper cross-encoder reranker (Hugging Face Transformers)."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .worker import InferenceWorker

log = logging.getLogger(__name__)


class Reranker:
    """Menilai relevansi pasangan (pertanyaan, passage) -> probabilitas 0..1."""

    def __init__(self, name: str, device: str = "cpu", max_length: int = 512, batch_size: int = 16) -> None:
        self.name = name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        t = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self._worker = InferenceWorker("reranker")  # satu thread GPU tetap, lihat worker.py
        log.info("Reranker %s (%s) dimuat dalam %.1fs", name, device, time.perf_counter() - t)

    def score(
        self, query: str, passages: Sequence[str], *, fp16: bool = False, batch_size: int | None = None
    ) -> np.ndarray:
        """Skor relevansi satu pertanyaan terhadap banyak passage (baseline: FP32, batch dari config)."""
        return self.score_pairs([(query, p) for p in passages], fp16=fp16, batch_size=batch_size)

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        fp16: bool = False,
        batch_size: int | None = None,
        sort_by_length: bool = False,
    ) -> np.ndarray:
        """Skor banyak pasangan (pertanyaan, passage) sekaligus — dipakai dynamic batching
        untuk menggabungkan kandidat dari beberapa permintaan dalam satu forward pass."""
        if not pairs:
            return np.empty(0, dtype=np.float32)
        bs = int(batch_size or self.batch_size)
        return self._worker.run(self._score, list(pairs), fp16, bs, sort_by_length)

    @torch.inference_mode()
    def _score(self, pairs: list[tuple[str, str]], fp16: bool, bs: int, sort_by_length: bool) -> np.ndarray:
        n = len(pairs)
        order = (
            sorted(range(n), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]), reverse=True)
            if sort_by_length
            else list(range(n))
        )
        out = np.empty(n, dtype=np.float32)
        use_fp16 = fp16 and self.device.startswith("cuda")
        for i in range(0, n, bs):
            idx = order[i : i + bs]
            enc = self.tokenizer(
                [pairs[j][0] for j in idx],
                [pairs[j][1] for j in idx],
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                logits = self.model(**enc).logits.float()
            if logits.shape[-1] == 1:
                prob = torch.sigmoid(logits[:, 0])
            else:  # model 2 kelas: ambil probabilitas kelas "relevan"
                prob = torch.softmax(logits, dim=-1)[:, -1]
            out[idx] = prob.cpu().numpy()
        return out
