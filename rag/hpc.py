"""Backend teroptimasi dari modul C++ ``rag_hpc`` (CUDA + OpenMP, lihat cpp/).

- ``cuda_topk``: pencarian vektor brute-force di GPU — kernel similarity (matriks FP16,
  akumulasi FP32) + seleksi top-k radix-select. Mendukung banyak kueri sekaligus
  (``search_batch``): matriks dibaca sekali per batch, sehingga batching hampir gratis.
- ``openmp``: BM25 paralel per segmen dokumen (tanpa atomik), top-k lokal lalu digabung.

Kedua backend mengembalikan hasil yang sama dengan backend baseline (``numpy`` /
``scipy``) sampai toleransi float; urutan dokumen dengan skor persis seri bisa berbeda.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import rag_hpc  # modul C++ hasil `bash cpp/build.sh`

from .index.bm25 import BM25Index

__version__ = rag_hpc.__version__


class CudaTopKBackend:
    name = "cuda_topk"
    supports_batch = True

    def __init__(self, embeddings: np.ndarray, threads: int = 0, dtype: str = "float16", device: int = 0) -> None:
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dim = int(emb.shape[1]) if emb.ndim == 2 else 0
        self.impl = rag_hpc.CudaTopK(emb, fp16=(dtype == "float16"), device=device)
        self.last_timings: dict[str, float] = {}

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx, scores, t = self.impl.search(np.asarray(q, dtype=np.float32).reshape(1, -1), int(k))
        self.last_timings = t
        return idx[0], scores[0]

    def search_batch(self, queries: np.ndarray, k: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, float]]:
        idx, scores, t = self.impl.search(np.ascontiguousarray(queries, dtype=np.float32), int(k))
        self.last_timings = t
        return [(idx[i], scores[i]) for i in range(idx.shape[0])], t


class OpenMPBM25Backend:
    name = "openmp"
    supports_batch = True

    def __init__(self, index: BM25Index, threads: int = 0, **_: Any) -> None:
        m = index.matrix
        indices = m.indices if m.indices.dtype == np.int32 else m.indices.astype(np.int32)
        self.vocab = index.vocab
        self.impl = rag_hpc.OpenMPBM25(m.data, indices, m.indptr.astype(np.int64), index.n_docs, int(threads))
        self.threads = self.impl.threads
        self.last_timings: dict[str, float] = {}

    def _cols(self, tokens: list[str]) -> np.ndarray:
        return np.asarray(sorted({self.vocab[t] for t in tokens if t in self.vocab}), dtype=np.int32)

    def search(self, tokens: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        idx, scores, t = self.impl.search(self._cols(tokens), int(k))
        self.last_timings = t
        return idx, scores

    def search_batch(self, token_lists: list[list[str]], k: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, float]]:
        out, tot = [], {"parallel_ms": 0.0, "merge_ms": 0.0}
        for toks in token_lists:  # tiap kueri memakai semua thread OpenMP
            idx, scores, t = self.impl.search(self._cols(toks), int(k))
            out.append((idx, scores))
            for key in tot:
                tot[key] += t.get(key, 0.0)
        return out, tot


def register(vector_backends: dict[str, Any], bm25_backends: dict[str, Any]) -> None:
    vector_backends[CudaTopKBackend.name] = CudaTopKBackend
    bm25_backends[OpenMPBM25Backend.name] = OpenMPBM25Backend


def info() -> dict[str, Any]:
    t = time.perf_counter()
    out: dict[str, Any] = {"version": __version__, "omp_max_threads": rag_hpc.omp_max_threads()}
    if rag_hpc.cuda_device_count() > 0:
        out["gpu"] = rag_hpc.device_info(0)
    out["probe_ms"] = round((time.perf_counter() - t) * 1000, 2)
    return out
