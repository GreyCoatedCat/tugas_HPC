"""Micro-benchmark kernel HPC (tanpa server, tanpa model): bahan analisis skalabilitas di laporan.

1. CUDA top-k (``rag_hpc.CudaTopK``) vs NumPy 1 thread (baseline)
   - N = 10^4, 10^5, 10^6 vektor (d=384), batch kueri m = 1, 4, 16, matriks FP16 & FP32;
   - waktu per kueri, waktu kernel similarity & radix-select (cudaEvent), dan **bandwidth efektif**
     kernel similarity (byte matriks / waktu kernel) dibandingkan bandwidth salin memori GPU
     (``torch`` device-to-device copy) sebagai batas atas praktis -> kernel memory-bound.
2. BM25 OpenMP — *strong scaling* pada indeks 10^6 chunk (koleksi dummy_l, 65 juta posting):
   speedup & efisiensi paralel untuk 1..16 thread, dibandingkan BM25 SciPy (baseline).

    python -m tools.bench_kernels                 # semua
    python -m tools.bench_kernels --skip-bm25     # tanpa indeks dummy_l

Hasil: results/kernels/kernels-<waktu>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.config import load_config
from rag.index.vector import NumpyBruteForce


def _normalized(n: int, d: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def gpu_copy_bandwidth_gbs(mb: int = 512, reps: int = 10) -> float | None:
    """Bandwidth salin device-to-device (baca + tulis) — batas atas praktis kernel memory-bound."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        x = torch.empty(mb * 2**20 // 4, dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)
        y.copy_(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            y.copy_(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        del x, y
        torch.cuda.empty_cache()
        return 2 * mb * 2**20 / dt / 1e9
    except Exception:  # noqa: BLE001
        return None


def bench_cuda(sizes: list[int], batches: list[int], reps: int, k: int) -> list[dict]:
    from rag.hpc import CudaTopKBackend

    d = 384
    rows = []
    for n in sizes:
        E = _normalized(n, d, 0)
        Q = _normalized(max(batches), d, 1)
        ref = NumpyBruteForce(E, threads=1)
        t = time.perf_counter()
        nq = 10 if n >= 10**6 else 30
        for i in range(nq):
            ref.search(Q[i % len(Q)], k)
        numpy_ms = (time.perf_counter() - t) / nq * 1000
        for dtype in ("float16", "float32"):
            g = CudaTopKBackend(E, dtype=dtype)
            g.search_batch(Q[:1], k)  # pemanasan
            for m in batches:
                walls, score, select = [], [], []
                for _ in range(reps):
                    t = time.perf_counter()
                    _, tm = g.search_batch(Q[:m], k)
                    walls.append((time.perf_counter() - t) * 1000)
                    score.append(tm["score_ms"])
                    select.append(tm["select_ms"])
                wall, sc, se = statistics.median(walls), statistics.median(score), statistics.median(select)
                bytes_read = n * d * (2 if dtype == "float16" else 4)
                rows.append({
                    "n": n, "dtype": dtype, "batch": m, "wall_ms": round(wall, 3), "per_query_ms": round(wall / m, 4),
                    "score_kernel_ms": round(sc, 4), "select_ms": round(se, 4),
                    "bandwidth_gbs": round(bytes_read / (sc / 1000) / 1e9, 1) if sc > 0 else None,
                    "numpy_1thread_ms": round(numpy_ms, 3), "speedup_vs_numpy": round(numpy_ms / (wall / m), 1),
                })
                r = rows[-1]
                print(f"[cuda] N={n:>8} {dtype:<7} m={m:>2}: {r['per_query_ms']:8.3f} ms/kueri | kernel {sc:7.3f} ms "
                      f"({r['bandwidth_gbs']} GB/s) | select {se:6.3f} ms | {r['speedup_vs_numpy']}x vs NumPy", flush=True)
            del g
        del E, ref
    return rows


def bench_bm25(threads: list[int], reps: int, n_queries: int) -> dict:
    from rag.hpc import OpenMPBM25Backend
    from rag.index.bm25 import BM25Index
    from rag.store import KnowledgeBase
    from rag.text import tokenize

    cfg = load_config()
    kb = KnowledgeBase(cfg["paths"]["kb_dir"], "dummy_l")
    idx = BM25Index.load(kb.index_dir)
    qs = [l.strip() for l in Path("data/dummy/dummy_l/questions.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    toks = [tokenize(q) for q in qs[:n_queries]]

    def per_query(fn) -> float:
        best = []
        for _ in range(reps):
            t = time.perf_counter()
            for tk in toks:
                fn(tk)
            best.append((time.perf_counter() - t) / len(toks) * 1000)
        return statistics.median(best)

    scipy_ms = per_query(lambda tk: idx.search(tk, 50))
    print(f"[bm25] scipy (baseline): {scipy_ms:.2f} ms/kueri — {idx.n_docs} dokumen, {idx.nnz} posting", flush=True)
    rows = []
    t1 = None
    for th in threads:
        b = OpenMPBM25Backend(idx, threads=th)
        b.search(toks[0], 50)
        ms = per_query(lambda tk: b.search(tk, 50))
        t1 = t1 or ms
        rows.append({"threads": th, "ms_per_query": round(ms, 3), "speedup_vs_1thread": round(t1 / ms, 2),
                     "efficiency": round(t1 / ms / th, 2), "speedup_vs_scipy": round(scipy_ms / ms, 1)})
        print(f"[bm25] {th:>2} thread: {ms:6.2f} ms/kueri | speedup {t1 / ms:5.2f}x (efisiensi {t1 / ms / th:4.0%}) | "
              f"{scipy_ms / ms:5.1f}x vs scipy", flush=True)
    return {"n_docs": idx.n_docs, "nnz": idx.nnz, "scipy_ms": round(scipy_ms, 3), "rows": rows}


def cpu_info() -> dict:
    info = {"logical_cpus": os.cpu_count()}
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10**4, 10**5, 10**6])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12, 16])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    a = ap.parse_args(argv)

    import rag_hpc

    report: dict = {"created": datetime.now().isoformat(timespec="seconds"), "cpu": cpu_info()}
    if rag_hpc.cuda_device_count() > 0:
        report["gpu"] = rag_hpc.device_info(0)
        report["gpu_copy_bandwidth_gbs"] = gpu_copy_bandwidth_gbs()
    if not a.skip_cuda:
        report["cuda_topk"] = bench_cuda(a.sizes, a.batches, a.reps, 50)
    if not a.skip_bm25:
        report["bm25_omp"] = bench_bm25(a.threads, 3, 40)

    L = [f"# Micro-benchmark kernel — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"CPU: {report['cpu'].get('model', '?')} ({report['cpu'].get('physical_cores', '?')} core fisik, "
         f"{report['cpu'].get('logical_cpus')} thread). GPU: {report.get('gpu', {}).get('name', '-')} "
         f"({report.get('gpu', {}).get('sm_count', '?')} SM). Bandwidth salin memori GPU (batas atas praktis): "
         f"**{report.get('gpu_copy_bandwidth_gbs') or 0:.0f} GB/s**.", ""]
    if "cuda_topk" in report:
        L += ["## CUDA top-k (similarity + radix-select), k = 50, d = 384", "",
              "| N | Matriks | Batch m | ms / kueri | Kernel similarity (ms) | Bandwidth efektif (GB/s) | Radix-select (ms) | vs NumPy 1 thread |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in report["cuda_topk"]:
            L.append(f"| {r['n']:,} | {r['dtype']} | {r['batch']} | {r['per_query_ms']:.3f} | {r['score_kernel_ms']:.3f} | "
                     f"{r['bandwidth_gbs']} | {r['select_ms']:.3f} | {r['speedup_vs_numpy']}× |")
        L += ["", "Kernel similarity membaca matriks sekali per batch, sehingga waktu per kueri turun hampir linear "
              "dengan ukuran batch (memory-bound). FP16 membaca separuh byte dibanding FP32.", ""]
    if "bm25_omp" in report:
        b = report["bm25_omp"]
        L += [f"## BM25 OpenMP — strong scaling ({b['n_docs']:,} dokumen, {b['nnz']:,} posting)", "",
              f"BM25 SciPy (baseline): {b['scipy_ms']:.2f} ms/kueri.", "",
              "| Thread | ms / kueri | Speedup vs 1 thread | Efisiensi | vs SciPy |", "|---:|---:|---:|---:|---:|"]
        for r in b["rows"]:
            L.append(f"| {r['threads']} | {r['ms_per_query']:.2f} | {r['speedup_vs_1thread']:.2f}× | "
                     f"{r['efficiency']:.0%} | {r['speedup_vs_scipy']}× |")
        L.append("")
    out = Path(load_config()["paths"]["results_dir"]) / "kernels"
    out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"kernels-{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"kernels-{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"Hasil: {out}/kernels-{tag}.md")


if __name__ == "__main__":
    main()
