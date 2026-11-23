"""Pencarian vektor brute-force (baseline: CPU, NumPy, 1 thread BLAS)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from .topk import topk

_controller = None


@contextmanager
def blas_threads(n: int | None) -> Iterator[None]:
    """Batasi jumlah thread BLAS (OpenBLAS/MKL) selama blok berjalan."""
    global _controller
    if not n or n <= 0:
        yield
        return
    if _controller is None:
        from threadpoolctl import ThreadpoolController

        _controller = ThreadpoolController()
    with _controller.limit(limits=n, user_api="blas"):
        yield


class NumpyBruteForce:
    """Exact search: skor = E @ q (cosine, karena vektor sudah dinormalisasi), lalu top-k.

    Kompleksitas O(N·d) per kueri. Ini adalah titik acuan untuk kernel CUDA
    kustom (similarity + top-k) di mode teroptimasi.
    """

    name = "numpy"

    def __init__(self, embeddings: np.ndarray, threads: int = 1, **_) -> None:
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.threads = threads

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.emb.shape[0] == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        with blas_threads(self.threads):
            scores = self.emb @ q
        return topk(scores, k)
