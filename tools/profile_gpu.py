"""Profiling GPU tingkat kernel untuk retrieval (torch.profiler / CUPTI).

Merekam SEMUA kernel CUDA di proses: kernel PyTorch (embedding, reranker: GEMM, attention,
softmax, ...) dan kernel C++ kita (``scores_kernel``, ``radix_hist_kernel``, ``gather_kernel``).
Setiap tahap pipeline diberi label (record_function) sehingga terlihat di timeline.

Keluaran di results/profiling/:
  - profile-<waktu>.md      tabel kernel/operator dengan waktu GPU terbesar per tahap
  - trace-<waktu>.json      timeline Chrome trace -> buka di https://ui.perfetto.dev atau chrome://tracing

    python -m tools.profile_gpu --collection dummy_l --mode optimized --batch 16
    python -m tools.profile_gpu --collection ugm --mode baseline --batch 1

Catatan: Nsight Systems/Compute dari paket Ubuntu (CUDA 12.0) belum mendukung GPU sm_120 dan
Nsight Compute butuh izin performance counter (root). torch.profiler membawa CUPTI CUDA 13.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from rag.index.hybrid import rrf_fuse
from rag.service import RAGService
from rag.text import tokenize


def questions_for(collection: str) -> list[str]:
    f = Path("data/dummy") / collection / "questions.txt"
    if not f.exists():
        f = Path("loadtest/questions_ugm.txt")
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def run_batch(svc: RAGService, mode, collection: str, qs: list[str]) -> None:
    """Replikasi tahap RAGService.retrieve_batch dengan label per tahap."""
    r = svc.retr
    kb, snap = svc.kb(collection), svc.snapshot(collection)
    emb, rer = svc.embedder(), svc.reranker
    vec = snap.vector_backend(mode.vector_backend, mode.vector_threads, mode.vector_dtype)
    bm = snap.bm25_backend(mode.bm25_backend, mode.bm25_threads)
    with record_function("1_embed_query"):
        Q = emb.encode(qs, batch_size=len(qs), is_query=True, fp16=mode.model_fp16, sort_by_length=False, overlap=False)
    with record_function("2_search_vector"):
        if getattr(vec, "supports_batch", False):
            v_res, _ = vec.search_batch(Q, int(r.get("top_k_vector", 50)))
        else:
            v_res = [vec.search(q, int(r.get("top_k_vector", 50))) for q in Q]
    with record_function("3_search_bm25"):
        b_res = [bm.search(tokenize(q), int(r.get("top_k_bm25", 50))) for q in qs]
    with record_function("4_fusion_fetch"):
        fused = [rrf_fuse([vi, bi], k=int(r.get("rrf_k", 60)), limit=int(r.get("rerank_candidates", 20)))
                 for (vi, _), (bi, _) in zip(v_res, b_res)]
        rows = kb.get_chunks(sorted({int(snap.chunk_ids[i]) for f in fused for i, _ in f}))
    with record_function("5_rerank"):
        pairs = [(q, f"{rows[int(snap.chunk_ids[i])]['title']}\n{rows[int(snap.chunk_ids[i])]['text']}")
                 for q, f in zip(qs, fused) for i, _ in f if int(snap.chunk_ids[i]) in rows]
        rer.score_pairs(pairs, fp16=mode.model_fp16, batch_size=mode.rerank_batch_size or None,
                        sort_by_length=mode.retrieval_batch_max > 1)
    torch.cuda.synchronize()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="dummy_l")
    ap.add_argument("--mode", default="optimized")
    ap.add_argument("--batch", type=int, default=16, help="jumlah pertanyaan per batch")
    ap.add_argument("--iters", type=int, default=3, help="batch yang direkam")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)

    svc = RAGService()
    svc.warmup(with_llm=False)
    mode = svc.get_mode(a.mode)
    qs = questions_for(a.collection)
    batches = [qs[(i * a.batch) % len(qs):][: a.batch] or qs[: a.batch] for i in range(a.iters + 1)]
    run_batch(svc, mode, a.collection, batches[0])  # pemanasan (salin matriks ke GPU, JIT kernel)

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for b in batches[1:]:
            run_batch(svc, mode, a.collection, b)
    wall = (time.perf_counter() - t0) * 1000

    out = Path(svc.cfg["paths"]["results_dir"]) / "profiling"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{datetime.now():%Y%m%d-%H%M%S}-{a.mode}-{a.collection}-b{a.batch}"
    trace = out / f"trace-{tag}.json"
    prof.export_chrome_trace(str(trace))

    ev = prof.key_averages()
    dev_time = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    stages = sorted([e for e in ev if e.key[:2] in {"1_", "2_", "3_", "4_", "5_"}], key=lambda e: e.key)
    kernels = sorted([e for e in ev if dev_time(e) > 0 and not e.key[:2] in {"1_", "2_", "3_", "4_", "5_"}
                      and e.key not in ("cudaDeviceSynchronize",)], key=dev_time, reverse=True)
    total_gpu = sum(dev_time(e) for e in ev if e.device_type is not None and str(e.device_type).endswith("CUDA")) or \
        sum(dev_time(e) for e in kernels)
    L = [f"# Profil GPU — mode `{a.mode}`, koleksi `{a.collection}`, batch {a.batch} — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"{a.iters} batch direkam, waktu dinding {wall:.0f} ms ({wall / a.iters:.1f} ms per batch, "
         f"{wall / a.iters / a.batch:.1f} ms per pertanyaan). Timeline: `{trace.name}` (buka di https://ui.perfetto.dev).", "",
         "## Per tahap", "",
         "Waktu dinding tahap di thread pemanggil (total semua batch). Kernel model dijalankan di thread inferensi "
         "khusus, jadi waktu GPU-nya tercantum di tabel kernel di bawah.", "",
         "| Tahap | Waktu dinding total (ms) | per batch (ms) |", "|---|---:|---:|"]
    agg: dict[str, float] = {}
    for e in stages:
        agg[e.key] = agg.get(e.key, 0.0) + e.cpu_time_total / 1000
    for k_, v in sorted(agg.items()):
        L.append(f"| {k_} | {v:.1f} | {v / a.iters:.1f} |")
    L += ["", f"## {a.top} kernel/operator GPU terbesar", "", "| Kernel / operator | Panggilan | Waktu GPU total (ms) | % |",
          "|---|---:|---:|---:|"]
    for e in kernels[: a.top]:
        name = e.key if len(e.key) <= 90 else e.key[:87] + "..."
        L.append(f"| `{name}` | {e.count} | {dev_time(e) / 1000:.2f} | {100 * dev_time(e) / max(total_gpu, 1):.1f}% |")
    L.append("")
    (out / f"profile-{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"Hasil: {out}/profile-{tag}.md")
    svc.close()


if __name__ == "__main__":
    main()
