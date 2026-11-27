"""Mode teroptimasi (CUDA/OpenMP/FP16) dan jalur dynamic batching harus memilih konteks yang sama
dengan baseline pada knowledge base resmi."""

import pytest

rag_hpc = pytest.importorskip("rag_hpc", reason="modul C++ belum dibangun (bash cpp/build.sh)")
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

from rag.service import RAGService  # noqa: E402
from rag.timing import StageTimer  # noqa: E402

QUESTIONS = [
    "Apa syarat mengajukan cuti akademik dan sampai kapan batas pengajuannya?",
    "Bagaimana prosedur pengajuan penurunan UKT?",
    "Berapa SKS maksimum yang boleh diambil jika IP semester sebelumnya di bawah 2,5?",
    "Kapan semester gasal 2026/2027 dimulai?",
    "Berapa lama masa studi maksimal program sarjana?",
    "Apa saja persyaratan beasiswa UGM?",
]


@pytest.fixture(scope="module")
def svc():
    import torch

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free < 1.5 * 2**30:
            pytest.skip(f"VRAM bebas hanya {free / 2**30:.1f} GB (butuh ±1,5 GB untuk embedder + reranker); "
                        "matikan demo / llama-server dulu")
    s = RAGService()
    snap = s.snapshot("ugm")
    if snap is None or snap.size == 0:
        pytest.skip("koleksi ugm belum diindeks (python -m scraper --ingest ugm)")
    yield s
    s.close()


def _top(hits, k=5):
    return [h["chunk_id"] for h in hits[:k]]


def test_optimized_matches_baseline_and_batch(svc):
    base, opt = svc.get_mode("baseline"), svc.get_mode("optimized")
    single = {}
    for q in QUESTIONS:
        hb, _ = svc.retrieve(q, "ugm", base, StageTimer())
        ho, _ = svc.retrieve(q, "ugm", opt, StageTimer())
        assert _top(hb) == _top(ho), q
        single[q] = _top(ho)
    timers = [StageTimer() for _ in QUESTIONS]
    results, _, info = svc.retrieve_batch(QUESTIONS, "ugm", opt, timers)
    assert info["batch_size"] == len(QUESTIONS)
    for q, hits in zip(QUESTIONS, results):
        assert _top(hits) == single[q], q
    assert all("rerank" in t.as_dict() for t in timers)
