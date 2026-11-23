"""CLI manajemen knowledge base & benchmark fase offline.

    python -m tools.kb list
    python -m tools.kb create NAMA [--description ...] [--synthetic]
    python -m tools.kb ingest NAMA FILE_ATAU_FOLDER... [--category peraturan] [--no-index]
    python -m tools.kb rebuild NAMA [--full] [--mode baseline]
    python -m tools.kb docs NAMA
    python -m tools.kb delete NAMA DOC_ID
    python -m tools.kb drop NAMA --yes
    python -m tools.kb ask NAMA "pertanyaan" [--no-generate]
    python -m tools.kb bench-build FILE_ATAU_FOLDER... [--repeats 3] [--mode baseline]
        -> ukur waktu fase offline (parse+clean, chunk, simpan, embedding, BM25) dari nol
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from rag.ingest import ingest_paths, warm_parse_pool
from rag.service import RAGService
from rag.store import KnowledgeBase


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_list(svc: RAGService, a) -> None:
    for st in svc.collections():
        idx = st["index"] or {}
        flag = " [SINTETIS]" if st["synthetic"] else ""
        stale = " (indeks perlu dibangun ulang)" if st["index_stale"] else ""
        print(f"- {st['name']}{flag}: {st['n_documents']} dokumen, {st['n_chunks']} chunk, "
              f"indeks={idx.get('n_chunks', 0)} chunk{stale}")


def cmd_create(svc: RAGService, a) -> None:
    _print_json(svc.create_collection(a.name, a.description, a.synthetic))


def cmd_ingest(svc: RAGService, a) -> None:
    kb = svc.kb(a.name)
    res = ingest_paths(kb, a.paths, chunk_cfg=svc.chunk_cfg, category=a.category, log_fn=print)
    for r in res["results"]:
        name = r.get("title") or r.get("filename") or r.get("path")
        extra = f"{r.get('n_chunks')} chunk" if r["status"] == "ok" else (r.get("reason") or r.get("error"))
        warn = f"  ⚠ {'; '.join(r['warnings'])}" if r.get("warnings") else ""
        print(f"  [{r['status']}] {name}: {extra}{warn}")
    print(f"Ingest: {res['ok']}/{res['files']} file, {res['n_chunks']} chunk, timing(ms)={res['timings_ms']}")
    if not a.no_index:
        meta = svc.rebuild(a.name, mode=a.mode, log_fn=print)
        print(f"Indeks: {meta['n_chunks']} chunk, timing(ms)={meta['timings_ms']}")


def cmd_rebuild(svc: RAGService, a) -> None:
    meta = svc.rebuild(a.name, full=a.full, mode=a.mode, log_fn=print)
    _print_json(meta)


def cmd_docs(svc: RAGService, a) -> None:
    for d in svc.kb(a.name).list_documents():
        print(f"{d['id']:>5}  [{d['category'] or '-':<10}] {d['title']}  ({d['n_chunks']} chunk)  {d['source_url'] or ''}")


def cmd_delete(svc: RAGService, a) -> None:
    n = svc.delete_document(a.name, a.doc_id)
    print(f"{n} chunk dihapus")
    if not a.no_index:
        svc.rebuild(a.name, log_fn=print)


def cmd_drop(svc: RAGService, a) -> None:
    if not a.yes:
        sys.exit("Tambahkan --yes untuk benar-benar menghapus koleksi beserta seluruh isinya.")
    KnowledgeBase.drop(svc.kb_root, a.name)
    print(f"Koleksi '{a.name}' dihapus.")


def cmd_ask(svc: RAGService, a) -> None:
    for ev in svc.answer_stream(a.question, collection=a.name, mode=a.mode, generate=not a.no_generate, client="cli"):
        if ev["type"] == "meta":
            print("Sumber:")
            for s in ev["sources"]:
                print(f"  [{s['n']}] {s['title']} (hlm {s['page_start']}) skor={s['rerank_score']}")
            print("\nJawaban:")
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "done":
            print(f"\n\nTTFT={ev['ttft_ms']} ms, total={ev['e2e_ms']} ms")
            _print_json(ev["timings_ms"])
        elif ev["type"] == "error":
            print(f"ERROR: {ev['message']}")


def cmd_bench_build(svc: RAGService, a) -> None:
    runs = []
    name = "bench-build-tmp"
    # muat model & pool worker lebih dulu agar waktu load/unduh/spawn tidak ikut terukur
    m = svc.get_mode(a.mode)
    svc.embedder().encode(["pemanasan", "model"], batch_size=2, fp16=(m.embed_dtype == "float16"))
    warm_parse_pool(m.ingest_workers)
    for r in range(a.repeats):
        KnowledgeBase.drop(svc.kb_root, name)
        svc._kbs.pop(name, None)  # noqa: SLF001
        svc.create_collection(name, "koleksi sementara untuk benchmark build", synthetic=True)
        kb = svc.kb(name)
        t0 = time.perf_counter()
        ing = ingest_paths(kb, a.paths, chunk_cfg=svc.chunk_cfg, workers=m.ingest_workers, log_fn=lambda *_: None)
        meta = svc.rebuild(name, full=True, mode=a.mode, log_fn=lambda *_: None)
        total = (time.perf_counter() - t0) * 1000
        # ingest_wall = waktu dinding; ingest_<tahap>_cpu = jumlah waktu CPU semua file (paralel: > dinding)
        t = {"ingest_wall": ing["timings_ms"]["total"],
             **{f"ingest_{k}_cpu": v for k, v in ing["timings_ms"].items() if k != "total"},
             **{f"index_{k}": v for k, v in meta["timings_ms"].items() if k != "total"}, "total": total}
        runs.append({"n_files": ing["files"], "n_chunks": meta["n_chunks"], "timings_ms": t})
        print(f"run {r + 1}/{a.repeats}: {ing['files']} file, {meta['n_chunks']} chunk, total {total / 1000:.2f}s")
    KnowledgeBase.drop(svc.kb_root, name)
    keys = list(runs[0]["timings_ms"])
    summary = {}
    print(f"\n{'tahap':<28}{'mean (ms)':>14}{'std (ms)':>12}")
    for k in keys:
        vals = [r["timings_ms"].get(k, 0.0) for r in runs]
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[k] = {"mean_ms": round(mean, 2), "std_ms": round(std, 2)}
        print(f"{k:<28}{mean:>14.1f}{std:>12.1f}")
    out_dir = svc.results_dir / "index_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"build-{a.mode}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"mode": a.mode, "paths": [str(p) for p in a.paths], "runs": runs, "summary": summary},
                              indent=2), encoding="utf-8")
    print(f"\nHasil: {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("create"); p.add_argument("name"); p.add_argument("--description", default=""); p.add_argument("--synthetic", action="store_true")
    p = sub.add_parser("ingest"); p.add_argument("name"); p.add_argument("paths", nargs="+"); p.add_argument("--category"); p.add_argument("--no-index", action="store_true"); p.add_argument("--mode", default="baseline")
    p = sub.add_parser("rebuild"); p.add_argument("name"); p.add_argument("--full", action="store_true"); p.add_argument("--mode", default="baseline")
    p = sub.add_parser("docs"); p.add_argument("name")
    p = sub.add_parser("delete"); p.add_argument("name"); p.add_argument("doc_id", type=int); p.add_argument("--no-index", action="store_true")
    p = sub.add_parser("drop"); p.add_argument("name"); p.add_argument("--yes", action="store_true")
    p = sub.add_parser("ask"); p.add_argument("name"); p.add_argument("question"); p.add_argument("--mode", default="baseline"); p.add_argument("--no-generate", action="store_true")
    p = sub.add_parser("bench-build"); p.add_argument("paths", nargs="+"); p.add_argument("--repeats", type=int, default=3); p.add_argument("--mode", default="baseline")
    a = ap.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.WARNING)
    svc = RAGService()
    {
        "list": cmd_list, "create": cmd_create, "ingest": cmd_ingest, "rebuild": cmd_rebuild, "docs": cmd_docs,
        "delete": cmd_delete, "drop": cmd_drop, "ask": cmd_ask, "bench-build": cmd_bench_build,
    }[a.cmd](svc, a)


if __name__ == "__main__":
    main()
