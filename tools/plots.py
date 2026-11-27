"""Grafik untuk laporan dari hasil di results/ -> results/figures/*.png

    python -m tools.plots                 # pakai hasil terbaru sejak hari ini
    python -m tools.plots --since 20261123 --llm-ref results/loadtest/20261120-120322_baseline_ugm

Grafik:
  fig1_retrieval_p95.png       p95 retrieval vs jumlah pengguna (KB resmi & 10^6 chunk)
  fig2_retrieval_qps.png       throughput retrieval vs jumlah pengguna
  fig3_stage_breakdown.png     komposisi latensi per tahap, 1 pengguna (baseline vs optimized)
  fig4_ablation.png            ablation kumulatif 10^6 chunk: latensi & throughput
  fig5_bm25_scaling.png        strong scaling BM25 OpenMP: speedup & efisiensi
  fig6_cuda_kernel.png         kernel CUDA: waktu per kueri vs batch & bandwidth efektif
  fig7_e2e_tokens.png          throughput token end-to-end dengan LLM
  fig8_build.png               waktu build indeks per tahap

Warna mengikuti entitas di semua grafik (baseline = biru, optimized = oranye, baseline Ollama
asli = hijau-biru); palet kategorikal tervalidasi untuk buta warna, setiap seri diberi label.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from rag.config import load_config  # noqa: E402
from tools.compare import by_users, latest_builds, latest_loadtests, token_stats  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # urutan tetap
BASE, OPT, REF = SERIES[0], SERIES[1], SERIES[2]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
        "font.size": 9, "lines.linewidth": 2, "lines.markersize": 6, "figure.dpi": 150,
    })


def _fmt_ms(v, _=None) -> str:
    return f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"


def _line(ax, xs, ys, color, label, marker="o", end_label=True):
    ax.plot(xs, ys, color=color, marker=marker, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    if end_label and xs:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)


def _save(fig, out: Path, name: str, note: str = "") -> Path:
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    p = out / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ", p)
    return p


def fig_retrieval(lt: dict, out: Path, metric: str, name: str, title: str, ylabel: str, log: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (col, sub) in zip(axes, (("ugm", "KB resmi UGM (1.920 chunk)"), ("dummy_l", "Korpus sintetis 10⁶ chunk"))):
        for mode, color, lbl in (("baseline", BASE, "Baseline"), ("optimized", OPT, "Optimized")):
            d = lt.get((mode, col, False))
            if not d:
                continue
            rows = by_users(d)
            xs = sorted(rows)
            _line(ax, xs, [rows[u][metric] for u in xs], color, lbl)
        ax.set_title(sub, loc="left")
        ax.set_xlabel("Pengguna konkuren")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 50, 100], ["1", "10", "50", "100"])
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))
        ax.margins(x=0.18)
        ax.legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, name, "Retrieval saja (tanpa LLM), uji beban Locust, rata-rata 3 pengulangan per titik.")


STAGE_GROUPS = [("search_vector", "Pencarian vektor"), ("search_bm25", "BM25"), ("rerank", "Rerank"),
                ("_other", "Embedding kueri + fusi + ambil chunk")]


def fig_stages(lt: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2))
    for ax, (col, sub) in zip(axes, (("ugm", "KB resmi UGM"), ("dummy_l", "10⁶ chunk"))):
        labels, rows = [], []
        for mode, lbl in (("optimized", "Optimized"), ("baseline", "Baseline")):
            d = lt.get((mode, col, False))
            if not d:
                continue
            st = by_users(d)[1]["stage_mean_ms"]
            other = sum(st.get(k, 0) for k in ("embed_query", "fusion", "fetch_chunks"))
            rows.append([st.get("search_vector", 0), st.get("search_bm25", 0), st.get("rerank", 0), other])
            labels.append(lbl)
        for y, vals in enumerate(rows):
            left = 0.0
            for (key, sname), v, color in zip(STAGE_GROUPS, vals, SERIES):
                ax.barh(y, v, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2,
                        label=sname if y == 0 else None)
                if v > sum(vals) * 0.08:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8, color="#ffffff")
                left += v
            ax.text(left, y, f"  {left:.0f} ms", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(sub, loc="left")
        ax.set_xlabel("Latensi rata-rata per permintaan, 1 pengguna (ms)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.12)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=4)
    fig.suptitle("Komposisi latensi retrieval per tahap", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, "fig3_stage_breakdown.png", "Angka di dalam batang = ms. Baseline: NumPy/SciPy/FP32; optimized: CUDA/OpenMP/FP16.")


def fig_ablation(root: Path, out: Path) -> None:
    files = sorted((root / "ablation").glob("ablation-*.json"))
    if not files:
        return
    rep = json.loads(files[-1].read_text())
    res = rep["collections"].get("dummy_l")
    if not res:
        return
    steps = [r["step"] for r in res]
    lat = [r["single_ms"]["total"] for r in res]
    qps = [r["throughput"]["qps"] for r in res]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, vals, ttl, unit in ((axes[0], lat, "Latensi 1 permintaan (ms)", "ms"),
                                (axes[1], qps, f"Throughput jenuh, {rep['users']} klien (QPS)", "")):
        bars = ax.bar(range(len(steps)), vals, color=OPT, width=0.6, edgecolor=SURFACE, linewidth=2)
        bars[0].set_color(BASE)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if unit else f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(steps)), steps, rotation=15)
        ax.set_title(ttl, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Ablation kumulatif — 10⁶ chunk (setiap batang menambah satu optimasi)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, "fig4_ablation.png", f"Sumber: {files[-1].name}. Biru = baseline, oranye = setelah optimasi.")


def _latest_kernels(root: Path) -> dict | None:
    files = sorted((root / "kernels").glob("kernels-*.json"))
    return json.loads(files[-1].read_text()) if files else None


def fig_bm25(root: Path, out: Path) -> None:
    rep = _latest_kernels(root)
    if not rep or "bm25_omp" not in rep:
        return
    rows = rep["bm25_omp"]["rows"]
    th = [r["threads"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(th, th, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[0].annotate("ideal (linear)", (th[-1], th[-1]), xytext=(-4, -2), textcoords="offset points", ha="right",
                     va="top", fontsize=8, color=INK2)
    _line(axes[0], th, [r["speedup_vs_1thread"] for r in rows], OPT, "BM25 OpenMP")
    axes[0].set_title("Speedup vs 1 thread", loc="left")
    axes[0].set_ylabel("Speedup (×)")
    _line(axes[1], th, [100 * r["efficiency"] for r in rows], OPT, "BM25 OpenMP")
    axes[1].set_title("Efisiensi paralel", loc="left")
    axes[1].set_ylabel("Efisiensi (%)")
    axes[1].set_ylim(0, 110)
    for ax in axes:
        ax.set_xlabel("Thread OpenMP")
        ax.set_xticks(th)
        ax.margins(x=0.15)
    phys = rep.get("cpu", {}).get("physical_cores")
    fig.suptitle(f"Strong scaling BM25 OpenMP — {rep['bm25_omp']['nnz'] / 1e6:.0f} juta posting, "
                 f"{rep['bm25_omp']['n_docs'] / 1e6:.0f} juta dokumen", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    _save(fig, out, "fig5_bm25_scaling.png", f"{rep['cpu'].get('model', '')} ({phys} core fisik: 6 P-core + 4 E-core). "
          f"BM25 SciPy baseline: {rep['bm25_omp']['scipy_ms']:.1f} ms/kueri.")


def fig_cuda(root: Path, out: Path) -> None:
    rep = _latest_kernels(root)
    if not rep or "cuda_topk" not in rep:
        return
    rows = rep["cuda_topk"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sizes = sorted({r["n"] for r in rows})
    for color, n in zip(SERIES, sizes):
        rs = sorted((r for r in rows if r["n"] == n and r["dtype"] == "float16"), key=lambda r: r["batch"])
        _line(axes[0], [r["batch"] for r in rs], [r["per_query_ms"] for r in rs], color, f"N = {n:,}".replace(",", "."))
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_fmt_ms))
    axes[0].set_xticks([1, 4, 16], ["1", "4", "16"])
    axes[0].set_xlabel("Ukuran batch kueri m")
    axes[0].set_ylabel("Waktu per kueri")
    axes[0].set_title("Batching kueri (matriks FP16)", loc="left")
    axes[0].margins(x=0.25)
    axes[0].legend(loc="lower left")
    for color, dt, lbl in ((OPT, "float16", "FP16"), (BASE, "float32", "FP32")):
        rs = sorted((r for r in rows if r["dtype"] == dt and r["batch"] == 1), key=lambda r: r["n"])
        _line(axes[1], [r["n"] for r in rs], [r["bandwidth_gbs"] for r in rs], color, lbl)
    bw = rep.get("gpu_copy_bandwidth_gbs")
    if bw:
        axes[1].axhline(bw, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
        axes[1].annotate(f"salin memori GPU (DRAM) {bw:.0f} GB/s", (sizes[-1], bw), xytext=(0, 5),
                         textcoords="offset points", ha="right", fontsize=8, color=INK2)
    peak = max([r["bandwidth_gbs"] or 0 for r in rows if r["batch"] == 1] + [bw or 0])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Jumlah vektor N")
    axes[1].set_ylabel("Bandwidth efektif kernel (GB/s)")
    axes[1].set_title("Kernel similarity, m = 1", loc="left")
    axes[1].set_ylim(0, peak * 1.18)
    axes[1].margins(x=0.2)
    axes[1].legend(loc="lower left")
    fig.suptitle(f"Kernel CUDA top-k — {rep.get('gpu', {}).get('name', 'GPU')}", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    _save(fig, out, "fig6_cuda_kernel.png",
          "d = 384, k = 50. Bandwidth efektif = byte matriks dibaca / waktu kernel similarity (cudaEvent). "
          "N = 10⁴ (15 MB) muat di cache L2 sehingga melampaui bandwidth DRAM.")


def fig_e2e(root: Path, lt: dict, llm_ref: str | None, out: Path) -> None:
    runs = []
    if llm_ref and (Path(llm_ref) / "summary.json").exists():
        runs.append(("Baseline asli (Ollama)", json.loads((Path(llm_ref) / "summary.json").read_text()), Path(llm_ref), REF))
    for mode, lbl, color in (("baseline", "Baseline (llama-server, serial)", BASE), ("optimized", "Optimized", OPT)):
        d = lt.get((mode, "ugm", True))
        if d:
            runs.append((lbl, d, root / "loadtest" / d["dir"], color))
    if len(runs) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for lbl, d, p, color in runs:
        ts = token_stats(p, d, root / "requests")
        xs = sorted(ts)
        _line(ax, xs, [ts[u]["tok_s"] for u in xs], color, lbl)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 50, 100], ["1", "10", "50", "100"])
    ax.set_xlabel("Pengguna konkuren")
    ax.set_ylabel("Token jawaban per detik (sistem)")
    ax.set_ylim(0, None)
    ax.margins(x=0.45)
    ax.legend(loc="upper left")
    ax.set_title("Throughput end-to-end dengan LLM — KB resmi UGM", loc="left", fontsize=12)
    _save(fig, out, "fig7_e2e_tokens.png", "Throughput token tidak bergantung panjang jawaban (parameter sampling Ollama dan llama.cpp berbeda).")


def fig_build(root: Path, since: str, out: Path) -> None:
    builds = latest_builds(root, since)
    items = []
    for src, lbl in (("raw", "14 dokumen resmi"), ("files_pdf_1000", "1.000 PDF sintetis")):
        for mode, ml in (("optimized", "optimized"), ("baseline", "baseline")):
            b = builds.get((mode, src))
            if b:
                s = b["summary"]
                ing = s["ingest_wall"]["mean_ms"] if "ingest_wall" in s else sum(
                    v["mean_ms"] for k, v in s.items() if k.startswith("ingest_"))
                emb = s["index_embed"]["mean_ms"]
                items.append((f"{lbl} — {ml}", ing / 1000, emb / 1000, max(0.0, s["total"]["mean_ms"] / 1000 - ing / 1000 - emb / 1000)))
    if not items:
        return
    fig, ax = plt.subplots(figsize=(9, 3.4))
    names = [("Parse + clean + chunk", SERIES[0]), ("Embedding", SERIES[1]), ("BM25 + simpan", SERIES[2])]
    for y, (lbl, *vals) in enumerate(items):
        left = 0.0
        for (sname, color), v in zip(names, vals):
            ax.barh(y, v, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2, label=sname if y == 0 else None)
            left += v
        ax.text(left, y, f"  {left:.1f} s", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.set_xlabel("Waktu build indeks dari nol (detik)")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    ax.set_title("Waktu build indeks per tahap", loc="left", fontsize=12, pad=26)
    _save(fig, out, "fig8_build.png", "Optimized: parsing paralel 8 proses (PDF tebal dipecah per 32 halaman), embedding batch 64 FP16.")


def main(argv=None) -> None:
    cfg = load_config()
    root = Path(cfg["paths"]["results_dir"])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--llm-ref", default="results/loadtest/20261120-120322_baseline_ugm")
    a = ap.parse_args(argv)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    lt = latest_loadtests(root, a.since)
    print("Menulis grafik ke", out)
    fig_retrieval(lt, out, "e2e_p95_ms", "fig1_retrieval_p95.png", "Latensi retrieval p95 vs jumlah pengguna",
                  "Latensi p95 (skala log)", log=True)
    fig_retrieval(lt, out, "qps", "fig2_retrieval_qps.png", "Throughput retrieval vs jumlah pengguna", "Permintaan selesai per detik", log=False)
    fig_stages(lt, out)
    fig_ablation(root, out)
    fig_bm25(root, out)
    fig_cuda(root, out)
    fig_e2e(root, lt, a.llm_ref, out)
    fig_build(root, a.since, out)


if __name__ == "__main__":
    main()
