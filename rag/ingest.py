"""Ingestion: file/teks bersih -> parsing -> cleaning -> chunking -> simpan ke KB.

Baseline (``workers=1``) memproses dokumen satu per satu. Mode teroptimasi
(``workers>1``) menjalankan parse + clean + chunk di pool proses paralel dengan
penjadwalan dinamis (file terbesar dulu / LPT), sementara proses utama menulis
hasilnya ke SQLite begitu tersedia (penulisan tetap satu penulis).

Paralelisme memakai proses, bukan thread OpenMP, karena parser PDF/HTML adalah
library Python yang tertahan GIL; polanya sama dengan ``omp parallel for
schedule(dynamic)`` atas daftar file.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from .loaders import (
    SUPPORTED_EXT,
    ParsedDocument,
    extract_pdf_text,
    parse_bytes,
    pdf_document_from_raw,
    pdf_info,
    sha256_bytes,
)
from .store import KnowledgeBase
from .text import chunk_pages
from .timing import StageTimer

log = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._\-]+", "_", name).strip("._") or "dokumen"
    return name[:120]


def ingest_parsed(
    kb: KnowledgeBase,
    parsed: ParsedDocument,
    *,
    doc_key: str,
    chunk_cfg: dict[str, int],
    source_url: str | None = None,
    category: str | None = None,
    file_path: str | None = None,
    sha256: str | None = None,
    extra_meta: dict | None = None,
    timer: StageTimer | None = None,
    chunks: list | None = None,
) -> dict[str, Any]:
    timer = timer or StageTimer()
    if chunks is None:
        with timer.stage("chunk"):
            chunks = chunk_pages(parsed.pages, **chunk_cfg)
    if not chunks:
        return {
            "status": "skipped",
            "title": parsed.title,
            "reason": "tidak ada teks yang bisa diekstrak",
            "warnings": parsed.warnings,
        }
    with timer.stage("store"):
        doc_id = kb.add_document(
            doc_key=doc_key,
            title=parsed.title,
            chunks=chunks,
            source_url=source_url,
            source_type=parsed.source_type,
            category=category,
            file_path=file_path,
            sha256=sha256,
            n_pages=len(parsed.pages),
            n_chars=parsed.n_chars,
            meta={**parsed.meta, **(extra_meta or {}), "warnings": parsed.warnings},
        )
    return {
        "status": "ok",
        "doc_id": doc_id,
        "title": parsed.title,
        "n_chunks": len(chunks),
        "n_chars": parsed.n_chars,
        "warnings": parsed.warnings,
    }


def ingest_bytes(
    kb: KnowledgeBase,
    data: bytes,
    filename: str,
    *,
    chunk_cfg: dict[str, int],
    title: str | None = None,
    category: str | None = None,
    source_url: str | None = None,
    content_type: str | None = None,
    store_file: bool = True,
) -> dict[str, Any]:
    timer = StageTimer()
    sha = sha256_bytes(data)
    with timer.stage("parse_clean"):
        parsed = parse_bytes(data, filename, content_type=content_type, url=source_url, title=title)
    file_path = None
    if store_file:
        dest = kb.files_dir / f"{sha[:12]}_{_safe_filename(filename)}"
        if not dest.exists():
            dest.write_bytes(data)
        file_path = str(dest)
    doc_key = f"url:{source_url}" if source_url else f"sha:{sha[:24]}"
    res = ingest_parsed(
        kb, parsed, doc_key=doc_key, chunk_cfg=chunk_cfg, source_url=source_url, category=category,
        file_path=file_path, sha256=sha, timer=timer,
    )
    res["filename"] = filename
    res["timings_ms"] = timer.as_dict()
    return res


def ingest_clean_json(kb: KnowledgeBase, path: str | Path, *, chunk_cfg: dict[str, int]) -> dict[str, Any]:
    """Ingest dokumen hasil scraper (data/scrape/clean/*.json)."""
    timer = StageTimer()
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    parsed = ParsedDocument(
        title=d["title"],
        pages=[(p.get("page"), p["text"]) for p in d.get("pages", [])],
        source_type=d.get("source_type", "html"),
        warnings=d.get("warnings", []),
        meta={"fetched_at": d.get("fetched_at"), "raw_path": d.get("raw_path")},
    )
    res = ingest_parsed(
        kb, parsed, doc_key=d.get("doc_key") or f"url:{d.get('url')}", chunk_cfg=chunk_cfg,
        source_url=d.get("url"), category=d.get("category"), sha256=d.get("sha256"), timer=timer,
    )
    res["path"] = str(path)
    res["timings_ms"] = timer.as_dict()
    return res


def iter_input_files(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and (f.suffix.lower() in SUPPORTED_EXT or f.suffix == ".json")))
        elif p.is_file():
            out.append(p)
    return out


# ---------------------------------------------------------------- paralel (mode teroptimasi)
_POOLS: dict[int, ProcessPoolExecutor] = {}


def _worker_init() -> None:
    from . import loaders, text  # noqa: F401  muat library parsing sekali per proses worker


def _noop(x: int) -> int:
    return x


def get_parse_pool(workers: int) -> ProcessPoolExecutor:
    pool = _POOLS.get(workers)
    if pool is None:
        # spawn: aman walau proses induk sudah memakai CUDA dan banyak thread
        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init)
        _POOLS[workers] = pool
    return pool


def warm_parse_pool(workers: int) -> None:
    """Hidupkan semua proses worker lebih dulu (dipakai sebelum pengukuran waktu build)."""
    if workers > 1:
        list(get_parse_pool(workers).map(_noop, range(workers * 2)))


def _prepare_file(path: str, chunk_cfg: dict[str, int], category: str | None, files_dir: str) -> dict[str, Any]:
    """Dijalankan di proses worker: baca + parse + clean + chunk (CPU-bound, tanpa akses SQLite)."""
    f = Path(path)
    timer = StageTimer()
    if f.suffix == ".json":
        d = json.loads(f.read_text(encoding="utf-8"))
        parsed = ParsedDocument(
            title=d["title"],
            pages=[(pg.get("page"), pg["text"]) for pg in d.get("pages", [])],
            source_type=d.get("source_type", "html"),
            warnings=d.get("warnings", []),
            meta={"fetched_at": d.get("fetched_at"), "raw_path": d.get("raw_path")},
        )
        out = {"doc_key": d.get("doc_key") or f"url:{d.get('url')}", "source_url": d.get("url"),
               "category": d.get("category"), "sha256": d.get("sha256"), "file_path": None}
    else:
        data = f.read_bytes()
        sha = sha256_bytes(data)
        with timer.stage("parse_clean"):
            parsed = parse_bytes(data, f.name)
        dest = Path(files_dir) / f"{sha[:12]}_{_safe_filename(f.name)}"
        if not dest.exists():
            dest.write_bytes(data)
        out = {"doc_key": f"sha:{sha[:24]}", "source_url": None, "category": category, "sha256": sha,
               "file_path": str(dest)}
    with timer.stage("chunk"):
        chunks = chunk_pages(parsed.pages, **chunk_cfg)
    return {**out, "parsed": parsed, "chunks": chunks, "name": f.name, "timings_ms": timer.as_dict()}


PAGES_PER_TASK = 32  # PDF lebih tebal dari ~1,5x ini dipecah per rentang halaman


def _extract_range(path: str, start: int, end: int) -> dict[str, Any]:
    """Worker: ekstraksi teks satu rentang halaman PDF (dekomposisi halus untuk PDF tebal)."""
    t = StageTimer()
    with t.stage("parse_clean"):
        raw = extract_pdf_text(path, start, end)
    return {"start": start, "raw": raw, "timings_ms": t.as_dict()}


def _finish_split_pdf(f: Path, parts: dict[int, dict], meta_title: str | None, chunk_cfg: dict[str, int],
                      category: str | None, files_dir: Path) -> dict[str, Any]:
    """Gabungkan rentang halaman (urut), lalu clean + chunk (murah dibanding ekstraksi)."""
    t = StageTimer()
    raw = [page for _, part in sorted(parts.items()) for page in part["raw"]]
    with t.stage("parse_clean"):
        parsed = pdf_document_from_raw(raw, meta_title, f.name)
    with t.stage("chunk"):
        chunks = chunk_pages(parsed.pages, **chunk_cfg)
    data = f.read_bytes()
    sha = sha256_bytes(data)
    dest = files_dir / f"{sha[:12]}_{_safe_filename(f.name)}"
    if not dest.exists():
        dest.write_bytes(data)
    timings = t.as_dict()
    timings["parse_clean"] = timings.get("parse_clean", 0.0) + sum(p_["timings_ms"].get("parse_clean", 0.0) for p_ in parts.values())
    return {"doc_key": f"sha:{sha[:24]}", "source_url": None, "category": category, "sha256": sha,
            "file_path": str(dest), "parsed": parsed, "chunks": chunks, "name": f.name, "timings_ms": timings}


def ingest_paths(
    kb: KnowledgeBase,
    paths: Iterable[str | Path],
    *,
    chunk_cfg: dict[str, int],
    category: str | None = None,
    workers: int = 1,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ingest banyak file. ``workers=1``: sekuensial (baseline). ``workers>1``: parse/clean/chunk
    paralel di pool proses, penyimpanan ke SQLite di proses utama secara bertahap.

    ``timings_ms``: per tahap = jumlah waktu CPU semua file; ``total`` = waktu dinding.
    """
    say = log_fn or log.info
    files = iter_input_files(paths)
    totals: dict[str, float] = {}
    results = []
    timer = StageTimer()

    def collect(res: dict[str, Any], i: int) -> None:
        for k, v in (res.get("timings_ms") or {}).items():
            totals[k] = totals.get(k, 0.0) + v
        results.append(res)
        if i % 50 == 0 or i == len(files):
            say(f"  ingest {i}/{len(files)}")

    if workers > 1 and len(files) > 0:
        pool = get_parse_pool(workers)
        # Dekomposisi tugas:
        # - PDF tebal dipecah per rentang halaman (PAGES_PER_TASK) -> beban satu buku tebal tersebar ke semua worker;
        # - file lain = satu tugas per file.
        # Tugas diurutkan dari yang terbesar (LPT) agar tidak ada worker yang selesai paling akhir sendirian.
        tasks: list[tuple[float, Any]] = []
        split: dict[Path, dict[str, Any]] = {}
        for f in files:
            n_pages, meta_title = 0, None
            if f.suffix.lower() == ".pdf":
                try:
                    n_pages, meta_title = pdf_info(f)
                except Exception:  # noqa: BLE001
                    n_pages = 0
            if n_pages > PAGES_PER_TASK * 1.5:
                ranges = [(s_, min(s_ + PAGES_PER_TASK, n_pages)) for s_ in range(0, n_pages, PAGES_PER_TASK)]
                split[f] = {"parts": {}, "n": len(ranges), "meta_title": meta_title}
                tasks += [(float(e - s_), ("range", f, s_, e)) for s_, e in ranges]
            else:
                # perkiraan biaya: ukuran berkas (~1 halaman PDF ≈ 15 KB)
                tasks.append((f.stat().st_size / 15e3, ("file", f)))
        tasks.sort(key=lambda x: -x[0])
        futs = {}
        for _, task in tasks:
            if task[0] == "range":
                _, f, s_, e = task
                futs[pool.submit(_extract_range, str(f), s_, e)] = task
            else:
                futs[pool.submit(_prepare_file, str(task[1]), chunk_cfg, category, str(kb.files_dir))] = task
        done = 0
        for fut in as_completed(futs):
            task = futs[fut]
            f = task[1]
            try:
                if task[0] == "range":
                    st = split[f]
                    st["parts"][task[2]] = fut.result()
                    if len(st["parts"]) < st["n"]:
                        continue  # tunggu rentang halaman lain dari PDF yang sama
                    prep = _finish_split_pdf(f, st["parts"], st["meta_title"], chunk_cfg, category, kb.files_dir)
                else:
                    prep = fut.result()
                t = StageTimer()
                res = ingest_parsed(
                    kb, prep["parsed"], doc_key=prep["doc_key"], chunk_cfg=chunk_cfg, source_url=prep["source_url"],
                    category=prep["category"], file_path=prep["file_path"], sha256=prep["sha256"], timer=t,
                    chunks=prep["chunks"],
                )
                res["filename"] = prep["name"]
                res["timings_ms"] = {**prep["timings_ms"], **t.as_dict()}
            except Exception as exc:  # noqa: BLE001
                if task[0] == "range" and split.get(f, {}).get("failed"):
                    continue
                if task[0] == "range":
                    split[f]["failed"] = True
                res = {"status": "error", "filename": f.name, "error": str(exc)}
            done += 1
            collect(res, done)
    else:
        for i, f in enumerate(files, 1):
            try:
                if f.suffix == ".json":
                    res = ingest_clean_json(kb, f, chunk_cfg=chunk_cfg)
                else:
                    res = ingest_bytes(kb, f.read_bytes(), f.name, chunk_cfg=chunk_cfg, category=category)
            except Exception as exc:  # noqa: BLE001
                res = {"status": "error", "filename": f.name, "error": str(exc)}
            collect(res, i)
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "files": len(files),
        "workers": workers,
        "ok": len(ok),
        "skipped": sum(r.get("status") == "skipped" for r in results),
        "errors": [r for r in results if r.get("status") == "error"],
        "n_chunks": sum(r.get("n_chunks", 0) for r in ok),
        "timings_ms": {**{k: round(v, 3) for k, v in totals.items()}, "total": round(timer.elapsed_ms(), 3)},
        "results": results,
    }
