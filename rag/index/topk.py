"""Seleksi top-k dari vektor skor."""

from __future__ import annotations

import numpy as np


def topk(scores: np.ndarray, k: int, positive_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n = int(scores.shape[0])
    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    k = min(k, n)
    idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    s = scores[idx]
    if positive_only:
        keep = s > 0
        idx, s = idx[keep], s[keep]
    return idx.astype(np.int64), s.astype(np.float32)
