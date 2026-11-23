"""Evaluasi kualitas retrieval (Recall@k) — memastikan optimasi tidak menurunkan kualitas.

QA file (JSONL) per baris minimal: {"question": ..., "title": ..., "answer"/"fact": ...}
Sebuah chunk dianggap relevan jika berasal dari dokumen berjudul `title` dan
memuat `fact` (atau `answer` bila fact tidak ada). Untuk QA buatan tangan pada
KB resmi, isi `title_contains` dan `must_contain` (daftar kata kunci).

    python -m tools.eval_recall --collection dummy_s --qa data/dummy/dummy_s/qa.jsonl
    python -m tools.eval_recall --collection ugm --qa data/eval/ugm_qa.jsonl --k 1 3 5 10
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.index.hybrid import rrf_fuse
from rag.service import RAGService
from rag.text import tokenize
from rag.timing import StageTimer


def is_relevant(row: dict, qa: dict) -> bool:
    text = " ".join(row["text"].split()).lower()
    title = (row.get("title") or "").lower()
    if qa.get("title") and title != qa["title"].lower():
        return False
    if qa.get("title_contains") and qa["title_contains"].lower() not in title:
        return False
    if qa.get("fact"):
        return " ".join(qa["fact"].split()).lower() in text
    if qa.get("must_contain"):
        return all(k.lower() in text for k in qa["must_contain"])
    if qa.get("answer"):
        return qa["answer"].lower() in text
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qa", required=True, type=Path)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah pertanyaan")
    a = ap.parse_args(argv)

    svc = RAGService()
    mode = svc.get_mode(a.mode)
    kb = svc.kb(a.collection)
    snap = svc.snapshot(a.collection)
    if snap is None:
        raise SystemExit("Koleksi belum diindeks.")
    if snap.meta.get("fake_embeddings"):
        print("PERINGATAN: indeks memakai vektor acak — Recall pencarian vektor tidak bermakna.")
    qas = [json.loads(l) for l in a.qa.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        qas = qas[: a.limit]
    kmax = max(a.k)
    r = svc.retr
    methods = ["vector", "bm25", "hybrid_rrf", "reranked"]
    hits_at = {m: {k: 0 for k in a.k} for m in methods}
    mrr = {m: [] for m in methods}
    stage_times: dict[str, list[float]] = {}

    for i, qa in enumerate(qas, 1):
        q = qa["question"]
        emb = svc.embedder().encode([q], is_query=True, fp16=mode.model_fp16)[0]
        v_idx, _ = snap.vector_backend(mode.vector_backend, mode.vector_threads, mode.vector_dtype).search(
            emb, max(kmax, int(r.get("top_k_vector", 50)))
        )
        b_idx, _ = snap.bm25_backend(mode.bm25_backend, mode.bm25_threads).search(
            tokenize(q), max(kmax, int(r.get("top_k_bm25", 50)))
        )
        fused = [i_ for i_, _ in rrf_fuse([v_idx, b_idx], k=int(r.get("rrf_k", 60)))]
        timer = StageTimer()
        reranked, _ = svc.retrieve(q, a.collection, mode, timer)
        for k_, v in timer.stages.items():
            stage_times.setdefault(k_, []).append(v)

        rankings = {
            "vector": [int(snap.chunk_ids[j]) for j in v_idx[:kmax]],
            "bm25": [int(snap.chunk_ids[j]) for j in b_idx[:kmax]],
            "hybrid_rrf": [int(snap.chunk_ids[j]) for j in fused[:kmax]],
            "reranked": [h["chunk_id"] for h in reranked[:kmax]],
        }
        all_ids = {c for lst in rankings.values() for c in lst}
        rows = kb.get_chunks(list(all_ids))
        rel = {c for c in all_ids if c in rows and is_relevant(rows[c], qa)}
        for m, lst in rankings.items():
            first = next((pos for pos, c in enumerate(lst, 1) if c in rel), None)
            mrr[m].append(1.0 / first if first else 0.0)
            for k in a.k:
                if first is not None and first <= k:
                    hits_at[m][k] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(qas)}")

    n = len(qas)
    print(f"\nRecall@k pada {n} pertanyaan (koleksi {a.collection}, mode {a.mode}, {snap.size} chunk)")
    header = f"{'metode':<12}" + "".join(f"{'R@' + str(k):>9}" for k in a.k) + f"{'MRR':>9}"
    print(header)
    result = {}
    for m in methods:
        rec = {k: hits_at[m][k] / n for k in a.k}
        result[m] = {"recall": {str(k): round(v, 4) for k, v in rec.items()}, "mrr": round(float(np.mean(mrr[m])), 4)}
        print(f"{m:<12}" + "".join(f"{rec[k]:>9.3f}" for k in a.k) + f"{result[m]['mrr']:>9.3f}")
    lat = {k: round(statistics.fmean(v), 2) for k, v in stage_times.items()}
    print(f"\nRata-rata latensi retrieval per tahap (ms): {lat}")
    out_dir = svc.results_dir / "recall"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"recall-{a.collection}-{a.mode}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"collection": a.collection, "mode": a.mode, "n_questions": n, "n_chunks": snap.size,
                               "results": result, "stage_mean_ms": lat}, indent=2), encoding="utf-8")
    print(f"Hasil: {out}")


if __name__ == "__main__":
    main()
