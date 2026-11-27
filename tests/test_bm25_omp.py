"""BM25 OpenMP harus menghasilkan skor & peringkat yang sama dengan BM25 SciPy (baseline)."""

import numpy as np
import pytest

rag_hpc = pytest.importorskip("rag_hpc", reason="modul C++ belum dibangun (bash cpp/build.sh)")

from rag.hpc import OpenMPBM25Backend  # noqa: E402
from rag.index.bm25 import BM25Index  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(7)
    # distribusi Zipf: ada term sangat umum (posting list panjang) dan term langka
    docs = [[f"t{min(int(v), 2000)}" for v in rng.zipf(1.3, size=int(rng.integers(20, 120)))] for _ in range(5000)]
    queries = [[f"t{int(v)}" for v in rng.zipf(1.5, size=int(rng.integers(2, 8)))] for _ in range(60)]
    return BM25Index.build(docs), queries


def _strictly_above(idx: np.ndarray, scores: np.ndarray, threshold: float) -> set[int]:
    return {int(i) for i, s in zip(idx, scores) if s > threshold + 1e-4}


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_matches_scipy(corpus, threads):
    index, queries = corpus
    omp = OpenMPBM25Backend(index, threads=threads)
    for q in queries:
        ref_idx, ref_sc = index.search(q, 50)
        idx, sc = omp.search(q, 50)
        assert len(idx) == len(ref_idx)
        np.testing.assert_allclose(sc, ref_sc, rtol=1e-5, atol=1e-5)
        # dokumen dengan skor persis seri di batas top-k boleh berbeda; di atas batas harus sama
        thr = float(ref_sc[-1]) if len(ref_sc) else 0.0
        assert _strictly_above(idx, sc, thr) == _strictly_above(ref_idx, ref_sc, thr)


def test_only_positive_scores_and_unknown_terms(corpus):
    index, _ = corpus
    omp = OpenMPBM25Backend(index, threads=4)
    idx, sc = omp.search(["istilah_yang_tidak_ada"], 10)
    assert idx.size == 0
    idx, sc = omp.search(["t1"], 10_000_000)
    assert np.all(sc > 0) and np.all(np.diff(sc) <= 1e-6)
