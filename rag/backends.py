"""Registry backend komputasi untuk tahap pencarian.

Mode di config.yaml merujuk backend lewat nama. Baseline memakai backend Python
(``numpy`` untuk vektor, ``scipy`` untuk BM25). Mode teroptimasi memakai backend
dari modul C++ ``rag_hpc`` (lihat ``rag/hpc.py`` dan ``cpp/``):

- ``cuda_topk``: kernel CUDA similarity + top-k (FP16, batch kueri);
- ``openmp``: BM25 paralel per segmen dokumen.

Backend didaftarkan otomatis bila ``rag_hpc*.so`` sudah dibangun (``bash cpp/build.sh``).
"""

from __future__ import annotations

import logging
from typing import Any

from .index.bm25 import BM25Index
from .index.vector import NumpyBruteForce

log = logging.getLogger(__name__)


class ScipyBM25:
    name = "scipy"

    def __init__(self, index: BM25Index, threads: int = 1, **_: Any) -> None:
        self.index = index

    def search(self, tokens, k):
        return self.index.search(tokens, k)


VECTOR_BACKENDS: dict[str, Any] = {"numpy": NumpyBruteForce}
BM25_BACKENDS: dict[str, Any] = {"scipy": ScipyBM25}
HPC_STATUS: dict[str, Any] = {"loaded": False, "detail": "modul rag_hpc belum dibangun"}


def _load_hpc_extension() -> None:
    try:
        from . import hpc  # mengimpor modul C++ rag_hpc (OpenMP/CUDA)
    except ImportError as exc:
        HPC_STATUS.update(loaded=False, detail=f"modul rag_hpc belum dibangun ({exc}); jalankan: bash cpp/build.sh")
        return
    try:
        hpc.register(VECTOR_BACKENDS, BM25_BACKENDS)
        HPC_STATUS.update(loaded=True, detail=f"rag_hpc {hpc.__version__}")
        log.info("Modul rag_hpc dimuat: vector=%s bm25=%s", list(VECTOR_BACKENDS), list(BM25_BACKENDS))
    except Exception as exc:  # noqa: BLE001
        HPC_STATUS.update(loaded=False, detail=f"gagal register rag_hpc: {exc}")
        log.warning(HPC_STATUS["detail"])


_load_hpc_extension()
