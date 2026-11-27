"""Kernel CUDA similarity + top-k harus setara dengan pencarian brute-force NumPy (baseline)."""

import numpy as np
import pytest

rag_hpc = pytest.importorskip("rag_hpc", reason="modul C++ belum dibangun (bash cpp/build.sh)")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(rag_hpc.cuda_device_count() == 0, reason="tidak ada GPU CUDA"),
]

from rag.hpc import CudaTopKBackend  # noqa: E402
from rag.index.vector import NumpyBruteForce  # noqa: E402


def _normalized(n: int, d: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("dtype, min_overlap, atol", [("float32", 1.0, 1e-5), ("float16", 0.98, 2e-4)])
def test_topk_matches_numpy(dtype, min_overlap, atol):
    E, Q = _normalized(20_000, 384, 0), _normalized(20, 384, 1)  # 20 kueri > max_batch 16 -> 2 potongan
    ref = NumpyBruteForce(E)
    gpu = CudaTopKBackend(E, dtype=dtype)
    results, timings = gpu.search_batch(Q, 50)
    assert len(results) == len(Q)
    assert timings["score_ms"] > 0
    for q, (idx, scores) in zip(Q, results):
        ref_idx, _ = ref.search(q, 50)
        assert len(set(idx.tolist()) & set(ref_idx.tolist())) / 50 >= min_overlap
        assert np.all(np.diff(scores) <= 1e-6), "skor harus terurut menurun"
        np.testing.assert_allclose(scores, E[idx] @ q, atol=atol)


def test_single_query_equals_batch():
    E, Q = _normalized(5_000, 384, 2), _normalized(3, 384, 3)
    gpu = CudaTopKBackend(E, dtype="float16")
    batch, _ = gpu.search_batch(Q, 10)
    for q, (idx_b, sc_b) in zip(Q, batch):
        idx_s, sc_s = gpu.search(q, 10)
        np.testing.assert_array_equal(idx_s, idx_b)
        np.testing.assert_allclose(sc_s, sc_b)


def test_k_larger_than_n_returns_all_rows_sorted():
    E = _normalized(7, 384, 4)
    idx, scores = CudaTopKBackend(E).search(E[0], 50)
    assert sorted(idx.tolist()) == list(range(7))
    assert idx[0] == 0  # vektor paling mirip dengan dirinya sendiri
    assert np.all(np.diff(scores) <= 1e-6)


def test_empty_matrix():
    idx, scores = CudaTopKBackend(np.zeros((0, 384), np.float32)).search(np.ones(384, np.float32), 5)
    assert idx.size == 0 and scores.size == 0
