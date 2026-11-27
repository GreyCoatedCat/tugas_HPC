"""Data untuk halaman "Hasil Speedup": ringkasan pengukuran sebelum (baseline) vs sesudah optimasi.

Semua angka dibaca dari berkas hasil di results/ (dibuat oleh loadtest/run_compare.sh, tools.compare,
tools.ablation, tools.bench_kernels, tools.eval_recall, tools.plots), sehingga halaman ikut berubah bila
pengukuran diulang. Bagian yang berkasnya belum ada dilewati.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.config import ROOT

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def _read(path: Path | None) -> str:
    try:
        return path.read_text(encoding="utf-8") if path else ""
    except OSError:
        return ""


def _latest(pattern: str) -> Path | None:
    files = sorted(RESULTS.glob(pattern))
    return files[-1] if files else None


def sections(md: str) -> dict[str, str]:
    """Pecah markdown per judul '## ' -> {judul: isi}."""
    out: dict[str, str] = {}
    title, buf = None, []
    for line in md.splitlines():
        if line.startswith("## "):
            if title is not None:
                out[title] = "\n".join(buf).strip()
            title, buf = line[3:].strip(), []
        elif title is not None:
            buf.append(line)
    if title is not None:
        out[title] = "\n".join(buf).strip()
    return out


def find_section(secs: dict[str, str], *keywords: str) -> str:
    for title, body in secs.items():
        if all(k.lower() in title.lower() for k in keywords):
            return body
    return ""


def tables(body: str) -> list[list[list[str]]]:
    """Semua tabel markdown di sebuah bagian; tiap tabel = [header, baris...] (baris pemisah dibuang)."""
    out, cur = [], []
    for line in body.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    return out


def num(cell: str) -> float | None:
    """'1,733' -> 1733.0, '6.8×' -> 6.8, '**129.3 ms**' -> 129.3."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", cell.replace("*", ""))
    return float(m.group(0).replace(",", "")) if m else None


def _id(v: float, digits: int = 1) -> str:
    """Format angka gaya Indonesia: 1.234,5."""
    s = f"{v:,.{digits}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def _secs(ms: float) -> str:
    return f"{_id(ms / 1000, 1 if ms >= 1000 else 2)} s"


def _col(header: list[str], *keywords: str) -> int | None:
    for i, h in enumerate(header):
        if all(k.lower() in h.lower() for k in keywords):
            return i
    return None


def _ratio(row: list[str], idx: int | None, fallback: float) -> float:
    """Pakai kolom percepatan yang sudah tertulis di tabel (dihitung dari angka tak dibulatkan) bila ada."""
    if idx is not None and idx < len(row) and "×" in row[idx]:
        v = num(row[idx])
        if v:
            return v
    return fallback


# ---------------------------------------------------------------------- ringkasan angka utama
def headline_cards(comparison_md: str) -> list[dict]:
    """Kartu angka besar: percepatan, nilai sebelum -> sesudah, dan konteksnya."""
    secs = sections(comparison_md)
    cards: list[dict] = []

    big = find_section(secs, "10⁶") or find_section(secs, "sintetis")
    tb = tables(big)
    if tb:
        h, rows = tb[0][0], tb[0][1:]
        u, qb, qo = _col(h, "Pengguna"), _col(h, "QPS baseline"), _col(h, "QPS optimized")
        pb, po = _col(h, "p95 baseline"), _col(h, "p95 optimized")
        tx, px = _col(h, "Throughput"), _col(h, "p95 lebih cepat")
        if None not in (u, qb, qo) and rows:
            r = max(rows, key=lambda r: num(r[u]) or 0)
            b, o = num(r[qb]), num(r[qo])
            if b and o:
                cards.append({"value": _ratio(r, tx, o / b), "label": "Throughput retrieval",
                              "detail": f"{_id(b, 1)} → {_id(o, 1)} kueri/detik · {r[u]} pengguna · 10⁶ chunk"})
        if None not in (u, pb, po) and rows:
            r = max(rows, key=lambda r: _ratio(r, px, (num(r[pb]) or 0) / (num(r[po]) or 1)))
            b, o = num(r[pb]), num(r[po])
            if b and o:
                cards.append({"value": _ratio(r, px, b / o), "label": "Latensi p95 lebih cepat",
                              "detail": f"{_secs(b)} → {_secs(o)} · {r[u]} pengguna · 10⁶ chunk"})
    if len(tb) > 1:
        stage = {r[0]: r for r in tb[1][1:]}
        for key, label, what in (("search_vector", "Pencarian vektor", "kernel CUDA top-k FP16"),
                                 ("search_bm25", "Pencarian BM25", "OpenMP C++")):
            r = stage.get(key)
            if r and num(r[1]) and num(r[2]):
                cards.append({"value": _ratio(r, 3, num(r[1]) / num(r[2])), "label": label,
                              "detail": f"{_id(num(r[1]))} → {_id(num(r[2]))} ms · {what}"})

    e2e = find_section(secs, "End-to-end")
    te = tables(e2e)
    if te:
        h, rows = te[0][0], te[0][1:]
        if len(h) >= 4 and rows:
            r = max(rows, key=lambda r: num(r[3]) or 0)  # beban dengan throughput optimized tertinggi
            b, o = num(r[1]), num(r[3])
            if b and o:
                cards.append({"value": o / b, "label": "Token jawaban per detik",
                              "detail": f"{_id(b)} → {_id(o)} token/s · {r[0]} pengguna · dengan LLM"})

    build = find_section(secs, "build")
    tbl = tables(build)
    if tbl:
        h, rows = tbl[0][0], tbl[0][1:]
        bi, oi, si = _col(h, "Baseline"), _col(h, "Optimized"), _col(h, "Percepatan")
        if None not in (bi, oi) and rows:
            r = max(rows, key=lambda r: _ratio(r, si, (num(r[bi]) or 0) / (num(r[oi]) or 1)))
            b, o = num(r[bi]), num(r[oi])
            if b and o:
                cards.append({"value": _ratio(r, si, b / o), "label": "Build indeks",
                              "detail": f"{_id(b)} → {_id(o)} s · {r[0]}"})
    return cards


def cards_html(cards: list[dict]) -> str:
    if not cards:
        return ""
    items = "".join(
        f'<div class="stat"><div class="stat-v">{_id(c["value"])}×</div><div class="stat-l">{c["label"]}</div>'
        f'<div class="stat-d">{c["detail"]}</div></div>'
        for c in cards
    )
    return f'<div class="stat-grid">{items}</div>'


# ---------------------------------------------------------------------- kualitas (Recall@k)
def quality_md() -> str:
    """Recall@k & MRR baseline vs optimized dari tools.eval_recall (run terbaru per mode)."""
    runs = {}
    for mode in ("baseline", "optimized"):
        p = _latest(f"recall/recall-*-{mode}-*.json")
        if p:
            try:
                runs[mode] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    if len(runs) < 2:
        return ""
    b, o = runs["baseline"], runs["optimized"]
    rows = ["| Tahap retrieval | Recall@1 baseline | Recall@1 optimized | Recall@5 baseline | Recall@5 optimized | "
            "MRR baseline | MRR optimized |", "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"vector": "Vektor", "bm25": "BM25", "hybrid_rrf": "Hybrid (RRF)", "reranked": "Setelah rerank"}
    for key, name in names.items():
        rb, ro = (b.get("results") or {}).get(key), (o.get("results") or {}).get(key)
        if not rb or not ro:
            continue
        rows.append(f"| {name} | {rb['recall']['1']:.3f} | {ro['recall']['1']:.3f} | {rb['recall']['5']:.3f} | "
                    f"{ro['recall']['5']:.3f} | {rb['mrr']:.3f} | {ro['mrr']:.3f} |")
    same = all((b.get("results") or {}).get(k) == (o.get("results") or {}).get(k) for k in names)
    verdict = ("**Identik.** Optimasi tidak mengubah dokumen yang ditemukan." if same
               else "**Berbeda** di sebagian metrik, cek urutan skor seri.")
    return (f"{verdict} {b.get('n_questions', '?')} pertanyaan uji pada koleksi `{b.get('collection', '?')}` "
            f"({_id(b.get('n_chunks', 0), 0)} chunk).\n\n" + "\n".join(rows))


# ---------------------------------------------------------------------- isi halaman
def _strip_title(md: str) -> str:
    return "\n".join(l for l in md.splitlines() if not l.startswith("# ")).strip()


def _subsections(md: str) -> str:
    """Turunkan '## ' menjadi '#### ' agar pas di dalam kartu halaman."""
    return re.sub(r"^## ", "#### ", md, flags=re.M)


def load() -> dict:
    comparison = _read(RESULTS / "comparison.md")
    secs = sections(comparison)
    m = re.search(r"results/ablation/(ablation-[\w-]+\.md)", comparison)
    ablation_path = (RESULTS / "ablation" / m.group(1)) if m else _latest("ablation/ablation-*.md")
    head = comparison.splitlines()[0][2:].strip() if comparison.startswith("# ") else ""
    return {
        "title": head,
        "cards": cards_html(headline_cards(comparison)),
        "big": find_section(secs, "10⁶"),
        "ugm": find_section(secs, "Retrieval saja", "UGM"),
        "e2e": find_section(secs, "End-to-end"),
        "build": find_section(secs, "build"),
        "llm_probe": find_section(secs, "Server LLM"),
        "ablation": _subsections(_strip_title(_read(ablation_path))),
        "ablation_name": ablation_path.name if ablation_path else "",
        "kernels": _subsections(_strip_title(_read(_latest("kernels/kernels-*.md")))),
        "quality": quality_md(),
        "figures": {p.stem: str(p) for p in sorted(FIGURES.glob("fig*.png"))},
    }
