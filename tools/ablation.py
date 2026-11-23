"""Ablation: kontribusi setiap optimasi retrieval, diterapkan satu per satu (kumulatif).

Konfigurasi (masing-masing menambah satu optimasi ke konfigurasi sebelumnya):
    0 baseline            NumPy 1 thread + scipy BM25 + model FP32, satu permintaan per waktu
    1 +cuda_topk          kernel CUDA similarity + radix-select top-k (matriks FP16)
    2 +bm25_openmp        BM25 paralel OpenMP (segmen dokumen per thread)
    3 +fp16               autocast FP16 untuk embedding kueri & reranker
    4 +batching (= optimized)  dynamic batching permintaan konkuren + overlap BM25 (CPU) || vektor (GPU)

Untuk tiap koleksi diukur (di dalam proses, tanpa HTTP, LLM tidak dipakai):
    - latensi per tahap satu permintaan (median dari --single permintaan);
    - throughput saat jenuh: --users thread klien tanpa jeda selama --duration detik.
Ringkasan juga memuat perhitungan hukum Amdahl untuk percepatan satu permintaan.

    python -m tools.ablation --collections ugm dummy_l
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rag.service import RAGService
from rag.timing import StageTimer

STEPS = [
    ("baseline", {}),
    ("+cuda_topk", {"vector_backend": "cuda_topk", "vector_dtype": "float16"}),
    ("+bm25_openmp", {"bm25_backend": "openmp", "bm25_threads": 8}),
    ("+fp16", {"model_fp16": True, "rerank_batch_size": 64}),
    ("+batching", {"retrieval_batch_max": 16, "retrieval_batch_window_ms": 0, "overlap_cpu_gpu": True}),
]


def questions_for(collection: str) -> list[str]:
    f = Path("data/dummy") / collection / "questions.txt"
    if not f.exists():
        f = Path("loadtest/questions_ugm.txt")
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def _one(svc: RAGService, mode, collection: str, q: str) -> dict[str, float]:
    t = StageTimer()
    t0 = time.perf_counter()
    svc.retrieve(q, collection, mode, t)
    d = t.as_dict()
    d["total"] = (time.perf_counter() - t0) * 1000
    return d


def single_latency_all(svc: RAGService, modes: list, collection: str, qs: list[str], n: int) -> list[dict[str, float]]:
    """Median latensi per tahap untuk setiap konfigurasi, diukur BERGILIRAN (round-robin):
    permintaan ke-i dijalankan di semua konfigurasi sebelum permintaan ke-i+1, sehingga drift
    termal GPU / cache tidak menguntungkan konfigurasi tertentu."""
    for i in range(n):  # pemanasan: cache halaman SQLite + backend tiap konfigurasi
        for m in modes:
            _one(svc, m, collection, qs[i % len(qs)])
    rows: list[list[dict[str, float]]] = [[] for _ in modes]
    for i in range(n):
        for j, m in enumerate(modes):
            rows[j].append(_one(svc, m, collection, qs[i % len(qs)]))
    out = []
    for r in rows:
        keys = r[0].keys()
        out.append({k: round(statistics.median(x.get(k, 0.0) for x in r), 3) for k in keys})
    return out


def throughput(svc: RAGService, mode_name: str, collection: str, qs: list[str], users: int, duration: float) -> dict:
    lat: list[float] = []
    lock = threading.Lock()
    stop = time.perf_counter() + duration

    def user(seed: int) -> None:
        rnd = random.Random(seed)
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            ok = False
            for ev in svc.answer_stream(rnd.choice(qs), collection=collection, mode=mode_name, generate=False,
                                        client="ablation"):
                ok = ok or ev["type"] == "done"
            if ok:
                with lock:
                    lat.append((time.perf_counter() - t0) * 1000)

    ths = [threading.Thread(target=user, args=(i,)) for i in range(users)]
    t0 = time.perf_counter()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    lat.sort()
    return {"qps": round(len(lat) / wall, 2), "p50_ms": round(lat[len(lat) // 2], 1) if lat else None,
            "p95_ms": round(lat[int(len(lat) * 0.95)], 1) if lat else None, "completed": len(lat)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collections", nargs="+", default=["ugm", "dummy_l"])
    ap.add_argument("--single", type=int, default=40, help="jumlah permintaan untuk median latensi tunggal")
    ap.add_argument("--users", type=int, default=32)
    ap.add_argument("--duration", type=float, default=20.0)
    a = ap.parse_args(argv)

    svc = RAGService()
    svc.warmup(with_llm=False)
    base = svc.get_mode("baseline")
    modes, cur = [], base
    for i, (label, change) in enumerate(STEPS):
        cur = replace(cur, name=f"abl{i}", description=label, **change)
        svc.register_mode(cur)
        modes.append((label, cur))

    report: dict = {"created": datetime.now().isoformat(timespec="seconds"), "users": a.users,
                    "duration_s": a.duration, "collections": {}}
    lines = [f"# Ablation optimasi retrieval — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Throughput: {a.users} klien tanpa jeda selama {a.duration:.0f} detik (di dalam proses, tanpa LLM). "
             f"Latensi tunggal: median {a.single} permintaan per konfigurasi, diukur bergiliran antar-konfigurasi "
             "setelah pemanasan (cache & backend hangat).", ""]
    for col in a.collections:
        qs = questions_for(col)
        svc.snapshot(col)
        res = []
        singles = single_latency_all(svc, [m for _, m in modes], col, qs, a.single)
        for (label, m), lat in zip(modes, singles):
            thr = throughput(svc, m.name, col, qs, a.users, a.duration)
            res.append({"step": label, "single_ms": lat, "throughput": thr})
            print(f"[{col}] {label:<14} tunggal {lat['total']:8.1f} ms | {thr['qps']:6.2f} QPS | "
                  f"p50 {thr['p50_ms']} ms | p95 {thr['p95_ms']} ms", flush=True)
        report["collections"][col] = res
        svc.unload(col)

        b = res[0]["single_ms"]
        stages = [k for k in b if k != "total"]
        lines += [f"## Koleksi `{col}`", "",
                  "| Konfigurasi | " + " | ".join(stages) + " | Total (ms) | Percepatan | QPS (jenuh) | p95 (ms) |",
                  "|---|" + "---:|" * (len(stages) + 4)]
        for r in res:
            s = r["single_ms"]
            lines.append(f"| {r['step']} | " + " | ".join(f"{s.get(k, 0):.1f}" for k in stages)
                         + f" | {s['total']:.1f} | {b['total'] / s['total']:.2f}× | {r['throughput']['qps']:.1f} | "
                           f"{r['throughput']['p95_ms']} |")
        o = res[-1]["single_ms"]
        lines += ["", "Hukum Amdahl (satu permintaan, baseline -> optimized):", "",
                  "| Tahap | Porsi baseline p | Percepatan lokal s | Waktu sesudah (ms) |", "|---|---:|---:|---:|"]
        for k in stages:
            p_ = b[k] / b["total"] if b["total"] else 0
            s_ = b[k] / o[k] if o.get(k) else float("inf")
            lines.append(f"| {k} | {p_:.1%} | {s_:.1f}× | {o.get(k, 0):.1f} |")
        sp = b["total"] / o["total"]
        lines += ["", f"Percepatan keseluruhan teramati: **{sp:.2f}×**. Tahap dengan porsi terbesar setelah optimasi: "
                  f"**{max(stages, key=lambda k: o.get(k, 0))}** — inilah bottleneck berikutnya.", ""]

    out = Path(svc.cfg["paths"]["results_dir"]) / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"ablation-{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"ablation-{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Hasil: {out}/ablation-{tag}.md")


if __name__ == "__main__":
    main()
