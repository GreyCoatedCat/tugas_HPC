"""Scraping + cleaning dokumen akademik UGM.

Contoh:
    python -m scraper                          # ambil semua seed, simpan raw + clean
    python -m scraper --ingest ugm             # sekaligus masukkan ke KB 'ugm' & bangun indeks
    python -m scraper --reclean                # ulangi cleaning dari file raw (tanpa download ulang)
    python -m scraper --only peraturan kalender

Output (default data/scrape/):
    raw/<host>/<hash>.<pdf|html>   file asli
    clean/<hash>.json              teks bersih per halaman + metadata + statistik cleaning
    manifest.jsonl                 log setiap fetch
    report.md                      ringkasan hasil scraping & cleaning
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import yaml

from rag.cleaning import text_stats
from rag.config import ROOT, load_config
from rag.loaders import parse_bytes, sha256_bytes

from .fetcher import FetchResult, PoliteFetcher


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def load_seeds(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [s for s in data.get("seeds", []) if s and s.get("url")]


def save_raw(res: FetchResult, raw_dir: Path) -> Path:
    host = urlparse(res.final_url or res.url).netloc or "unknown"
    ext = "pdf" if res.is_pdf else ("html" if res.is_html else "bin")
    d = raw_dir / host
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{url_hash(res.url)}.{ext}"
    p.write_bytes(res.content)
    return p


def clean_document(
    data: bytes, *, url: str, content_type: str, title: str | None, category: str | None,
    raw_path: Path, fetched_at: str, final_url: str | None = None,
) -> dict:
    parsed = parse_bytes(data, Path(urlparse(url).path).name or "index.html", content_type=content_type, url=url, title=title)
    clean_text = parsed.text
    st = text_stats(clean_text)
    return {
        "doc_key": f"url:{url}",
        "url": url,
        "final_url": final_url or url,
        "title": parsed.title,
        "category": category,
        "source_type": parsed.source_type,
        "fetched_at": fetched_at,
        "sha256": sha256_bytes(data),
        "raw_path": str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path),
        "pages": [{"page": p, "text": t} for p, t in parsed.pages],
        "stats": {
            "raw_chars": parsed.meta.get("raw_chars"),
            "clean_chars": st["chars"],
            "words": st["words"],
            "alpha_ratio": st["alpha_ratio"],
            "n_pages": len(parsed.pages),
            "n_pages_total": parsed.meta.get("n_pages_total"),
        },
        "warnings": parsed.warnings,
    }


def extract_links(html: bytes, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = urldefrag(urljoin(base_url, a["href"].strip()))[0]
        if href.startswith(("http://", "https://")):
            out.append(href)
    return list(dict.fromkeys(out))


def crawl(seeds: list[dict], out_dir: Path, fetcher: PoliteFetcher, log=print) -> list[dict]:
    raw_dir, clean_dir = out_dir / "raw", out_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    visited: set[str] = set()
    rows: list[dict] = []

    for seed in seeds:
        crawl_cfg = seed.get("crawl") or {}
        max_depth = int(crawl_cfg.get("depth", 0))
        max_pages = int(crawl_cfg.get("max_pages", 1 if not crawl_cfg else 20))
        allow = [re.compile(p) for p in crawl_cfg.get("allow", [])]
        follow_pdf = bool(crawl_cfg.get("follow_pdf", True))
        seed_host = urlparse(seed["url"]).netloc
        queue = deque([(seed["url"], 0)])
        n_pages = 0
        while queue and n_pages < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            n_pages += 1
            log(f"[fetch] {url}")
            res = fetcher.fetch(url)
            entry = {
                "url": url, "final_url": res.final_url, "status": res.status, "content_type": res.content_type,
                "bytes": len(res.content), "fetched_at": res.fetched_at, "error": res.error,
                "seed": seed["url"], "category": seed.get("category"),
                "title": seed.get("title") if depth == 0 else None,
            }
            row = {"url": url, "category": seed.get("category"), "status": "error", "error": res.error}
            if res.ok and (res.is_pdf or res.is_html):
                raw_path = save_raw(res, raw_dir)
                entry["raw_path"] = str(raw_path)
                try:
                    doc = clean_document(
                        res.content, url=url, content_type=res.content_type, title=entry["title"],
                        category=seed.get("category"), raw_path=raw_path, fetched_at=res.fetched_at,
                        final_url=res.final_url,
                    )
                    (clean_dir / f"{url_hash(url)}.json").write_text(
                        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
                    row.update(status="ok", error=None, title=doc["title"], type=doc["source_type"], **doc["stats"],
                               warnings="; ".join(doc["warnings"]))
                except Exception as exc:  # noqa: BLE001
                    row.update(error=f"cleaning gagal: {exc}")
                if res.is_html and depth < max_depth:
                    for link in extract_links(res.content, res.final_url or url):
                        if link in visited:
                            continue
                        is_pdf = link.lower().split("?")[0].endswith(".pdf")
                        host = urlparse(link).netloc
                        same_host = host == seed_host
                        official = host == "ugm.ac.id" or host.endswith(".ugm.ac.id")
                        if is_pdf and follow_pdf and official:
                            queue.append((link, depth + 1))
                        elif same_host and (not allow or any(p.search(link) for p in allow)):
                            queue.append((link, depth + 1))
            elif res.ok:
                row["error"] = f"tipe konten tidak didukung: {res.content_type}"
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            rows.append(row)
            log(f"   -> {row['status']}" + (f" ({row['error']})" if row.get("error") else f" {row.get('n_pages')} hlm, {row.get('clean_chars')} karakter"))
    return rows


def reclean(out_dir: Path, seeds: list[dict] | None = None, log=print) -> list[dict]:
    """Ulangi cleaning dari file raw berdasarkan manifest (entri terbaru per URL).

    Judul & kategori diambil dari seeds.yaml bila URL-nya terdaftar di sana.
    """
    seed_map = {s["url"]: s for s in (seeds or [])}
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit("manifest.jsonl belum ada — jalankan scraping dulu.")
    latest: dict[str, dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("raw_path") and not e.get("error"):
            latest[e["url"]] = e
    rows = []
    for url, e in latest.items():
        raw_path = Path(e["raw_path"])
        if not raw_path.exists():
            continue
        seed = seed_map.get(url, {})
        doc = clean_document(
            raw_path.read_bytes(), url=url, content_type=e.get("content_type", ""),
            title=seed.get("title") or e.get("title"), category=seed.get("category") or e.get("category"),
            raw_path=raw_path, fetched_at=e["fetched_at"], final_url=e.get("final_url"),
        )
        (out_dir / "clean" / f"{url_hash(url)}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append({"url": url, "category": e.get("category"), "status": "ok", "title": doc["title"],
                     "type": doc["source_type"], **doc["stats"], "warnings": "; ".join(doc["warnings"])})
        log(f"[reclean] {doc['title']}: {doc['stats']['clean_chars']} karakter")
    return rows


def write_report(rows: list[dict], path: Path) -> None:
    lines = [
        f"# Laporan scraping — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| Status | Kategori | Judul / URL | Tipe | Halaman | Karakter raw | Karakter bersih | Alpha ratio | Catatan |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = r.get("title") or r["url"]
        lines.append(
            f"| {r['status']} | {r.get('category') or ''} | [{name}]({r['url']}) | {r.get('type', '')} | "
            f"{r.get('n_pages', '')} | {r.get('raw_chars', '')} | {r.get('clean_chars', '')} | "
            f"{r.get('alpha_ratio', '')} | {r.get('error') or r.get('warnings') or ''} |"
        )
    ok = sum(r["status"] == "ok" for r in rows)
    lines += ["", f"Berhasil: {ok}/{len(rows)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ingest_into(collection: str, clean_files: list[Path], full_rebuild: bool, log=print) -> None:
    from rag.ingest import ingest_clean_json
    from rag.service import RAGService

    svc = RAGService()
    try:
        kb = svc.kb(collection)
    except FileNotFoundError:
        svc.create_collection(collection, "Dokumen akademik resmi UGM (hasil scraping)")
        kb = svc.kb(collection)
    if kb.synthetic:
        raise SystemExit(f"Koleksi '{collection}' bertanda synthetic — dokumen resmi tidak boleh dicampur.")
    for f in clean_files:
        res = ingest_clean_json(kb, f, chunk_cfg=svc.chunk_cfg)
        log(f"[ingest] {res.get('title')}: {res['status']} {res.get('n_chunks', '')} chunk")
    meta = svc.rebuild(collection, full=full_rebuild, log_fn=log)
    log(f"[index] {meta['n_chunks']} chunk terindeks, waktu: {meta['timings_ms']}")


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=ROOT / "scraper" / "seeds.yaml")
    ap.add_argument("--out", type=Path, default=Path(cfg["paths"]["scrape_dir"]))
    ap.add_argument("--only", nargs="*", help="hanya kategori tertentu")
    ap.add_argument("--url", nargs="*", help="scrape URL tertentu saja (tanpa seeds.yaml)")
    ap.add_argument("--delay", type=float, default=1.5, help="jeda antar request ke host yang sama (detik)")
    ap.add_argument("--reclean", action="store_true", help="ulangi cleaning dari raw tanpa download")
    ap.add_argument("--ingest", metavar="KOLEKSI", help="masukkan hasil ke koleksi KB lalu bangun indeks")
    ap.add_argument("--full-rebuild", action="store_true", help="hitung ulang seluruh embedding saat ingest")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    all_seeds = load_seeds(args.seeds) if args.seeds.exists() else []
    if args.reclean:
        rows = reclean(args.out, all_seeds)
    else:
        by_url = {s["url"]: s for s in all_seeds}
        seeds = [by_url.get(u, {"url": u}) for u in args.url] if args.url else all_seeds
        if args.only:
            seeds = [s for s in seeds if s.get("category") in set(args.only)]
        if not seeds:
            sys.exit("Tidak ada seed yang cocok.")
        rows = crawl(seeds, args.out, PoliteFetcher(delay_s=args.delay))
    write_report(rows, args.out / "report.md")
    print(f"\nLaporan: {args.out / 'report.md'}")

    if args.ingest:
        files = [args.out / "clean" / f"{url_hash(r['url'])}.json" for r in rows if r["status"] == "ok"]
        ingest_into(args.ingest, [f for f in files if f.exists()], args.full_rebuild)


if __name__ == "__main__":
    main()
