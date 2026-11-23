"""Penggabungan hasil pencarian vektor + BM25 dengan Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rrf_fuse(rankings: Sequence[np.ndarray], k: int = 60, limit: int | None = None) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            i = int(idx)
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    items = sorted(scores.items(), key=lambda x: -x[1])
    return items[:limit] if limit else items
