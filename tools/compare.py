"""Laporan perbandingan sebelum vs sesudah optimasi -> results/comparison.md.

Mengambil hasil TERBARU per (mode, koleksi, jenis uji) dari:
    results/loadtest/*/summary.json      uji beban retrieval & end-to-end
    results/index_build/build-*.json     waktu build indeks
    results/llm_probe/*.json             kinerja server LLM
    results/ablation/ablation-*.md       kontribusi per optimasi

    python -m tools.compare --since 20261121
    python -m tools.compare --llm-ref results/loadtest/20261120-120322_baseline_ugm   # baseline Ollama asli
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from rag.config import load_config


def latest_loadtests(root: Path, since: str) -> dict[tuple[str, str, bool], dict]:
    out: dict[tuple[str, str, bool], tuple[str, dict]] = {}
    for d in sorted((root / "loadtest").glob("2*")):
        f = d / "summary.json"
        if not f.exists() or d.name[:8] < since:
            continue
        data = json.loads(f.read_text())
        c = data["config"]
        key = (c["mode"], c.get("collection") or "ugm", bool(c.get("generate", True)))
        out[key] = (d.name, data)  # urut nama = urut waktu -> yang terakhir menang
    return {k: {"dir": v[0], **v[1]} for k, v in out.items()}


def by_users(data: dict) -> dict[int, dict]:
    return {int(r["users"]): r for r in data["summary"]}


def ratio(a, b, inverse=False) -> str:
    if a in (None, 0) or b in (None, 0):
        return "-"
    r = (b / a) if inverse else (a / b)
    return f"{r:.1f}×"


def f0(v, nd=0) -> str:
    return "-" if v is None else f"{v:,.{nd}f}"


def retrieval_table(b: dict, o: dict) -> list[str]:
    bu, ou = by_users(b), by_users(o)
    lines = ["| Pengguna | QPS baseline | QPS optimized | Throughput | p50 baseline (ms) | p50 optimized (ms) | "
             "p95 baseline (ms) | p95 optimized (ms) | p95 lebih cepat |", "|---:|" + "---:|" * 8]
    for u in sorted(set(bu) & set(ou)):
        x, y = bu[u], ou[u]
        lines.append(f"| {u} | {x['qps']:.2f} | {y['qps']:.2f} | {ratio(x['qps'], y['qps'], inverse=True)} | "
                     f"{f0(x['e2e_p50_ms'])} | {f0(y['e2e_p50_ms'])} | {f0(x['e2e_p95_ms'])} | {f0(y['e2e_p95_ms'])} | "
                     f"{ratio(x['e2e_p95_ms'], y['e2e_p95_ms'])} |")
    return lines


def stage_table(b: dict, o: dict, users: int) -> list[str]:
    x, y = by_users(b).get(users), by_users(o).get(users)
    if not x or not y:
        return []
    keys = [k for k in x["stage_mean_ms"] if not k.startswith("server_e2e")]
    lines = [f"Rata-rata per tahap pada {users} pengguna (ms):", "", "| Tahap | Baseline | Optimized | Percepatan |",
             "|---|---:|---:|---:|"]
    for k in keys:
        a, c = x["stage_mean_ms"].get(k), y["stage_mean_ms"].get(k)
        lines.append(f"| {k} | {f0(a, 1)} | {f0(c, 1)} | {ratio(a, c)} |")
    return lines


def token_stats(run_dir: Path, data: dict, req_dir: Path) -> dict[int, dict]:
    """Token per jawaban & throughput token/detik per tingkat pengguna, dari log request server.

    Jendela waktu tiap run direkonstruksi dari nama folder (waktu mulai) + wall_s + cooldown.
    Metrik ini tidak bergantung panjang jawaban, sehingga adil untuk membandingkan runtime LLM
    dengan parameter sampling berbeda (jawaban lebih panjang = QPS lebih rendah).
    """
    name = run_dir.name
    start = datetime.strptime(name[:15], "%Y%m%d-%H%M%S").timestamp()
    mode = data["config"]["mode"]
    cooldown = float(data["config"].get("cooldown", 5))
    recs: list[tuple[float, int]] = []
    log = req_dir / f"requests-{name[:8]}.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            c = (r.get("llm") or {}).get("completion_tokens")
            if r.get("client") == "locust" and r.get("mode") == mode and not r.get("error") and c:
                recs.append((float(r["ts"]), int(c)))
    acc: dict[int, dict] = {}
    t = start
    for run in data["runs"]:
        end = t + float(run["wall_s"])
        toks = [c for ts, c in recs if t <= ts <= end + 1.0]
        a = acc.setdefault(int(run["users"]), {"tokens": 0, "wall": 0.0, "answers": []})
        a["tokens"] += sum(toks)
        a["wall"] += float(run["wall_s"])
        a["answers"] += toks
        t = end + cooldown
    out = {}
    for u, a in acc.items():
        ans = sorted(a["answers"])
        out[u] = {"tok_s": a["tokens"] / a["wall"] if a["wall"] else None,
                  "tok_per_answer": ans[len(ans) // 2] if ans else None}
    return out


def e2e_tables(runs: list[tuple[str, dict, Path]], req_dir: Path) -> list[str]:
    users = sorted(set.intersection(*[set(by_users(d)) for _, d, _ in runs]))
    toks = [token_stats(p, d, req_dir) for _, d, p in runs]
    labels = [lbl for lbl, _, _ in runs]
    hdr = "| Pengguna | " + " | ".join(labels) + " |"
    sep = "|---:|" + "---:|" * len(runs)
    L = ["**Throughput token** (token jawaban yang dihasilkan sistem per detik — tidak bergantung panjang jawaban):", "",
         hdr, sep]
    for u in users:
        L.append(f"| {u} | " + " | ".join(f0(t.get(u, {}).get("tok_s"), 1) for t in toks) + " |")
    L += ["", "**Jawaban selesai per detik (QPS)** dan jumlah selesai / ditinggal per run:", "", hdr, sep]
    for u in users:
        cells = []
        for _, d, _ in runs:
            r = by_users(d)[u]
            cells.append(f"{r['qps']:.3f} ({f0(r['completed'])} / {f0(r.get('abandoned'))})")
        L.append(f"| {u} | " + " | ".join(cells) + " |")
    L += ["", "**TTFT p50 / E2E p50 (detik)** — hanya dari permintaan yang selesai:", "", hdr, sep]
    for u in users:
        cells = []
        for _, d, _ in runs:
            r = by_users(d)[u]
            cells.append(f"{f0((r['ttft_p50_ms'] or 0) / 1000, 1)} / {f0((r['e2e_p50_ms'] or 0) / 1000, 1)}")
        L.append(f"| {u} | " + " | ".join(cells) + " |")
    L += ["", "**Median token per jawaban:** " + " · ".join(
        f"{lbl}: {f0(statistics.median([v['tok_per_answer'] for v in t.values() if v.get('tok_per_answer')]) if any(v.get('tok_per_answer') for v in t.values()) else None)}"
        for lbl, t in zip(labels, toks)), ""]
    return L


def latest_builds(root: Path, since: str) -> dict[tuple[str, str], dict]:
    out = {}
    for f in sorted((root / "index_build").glob("build-*.json")):
        stamp = f.stem.split("-")[-2]
        if stamp < since:
            continue
        d = json.loads(f.read_text())
        out[(d["mode"], Path(d["paths"][0]).name)] = d
    return out


def main(argv=None) -> None:
    cfg = load_config()
    root = Path(cfg["paths"]["results_dir"])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=datetime.now().strftime("%Y%m%d"), help="hanya hasil sejak YYYYMMDD")
    ap.add_argument("--llm-ref", default=None, help="folder hasil end-to-end baseline asli (Ollama bawaan)")
    ap.add_argument("--out", default=str(root / "comparison.md"))
    a = ap.parse_args(argv)

    lt = latest_loadtests(root, a.since)
    L = [f"# Perbandingan sebelum vs sesudah optimasi — {datetime.now():%Y-%m-%d %H:%M}", "",
         "Sumber: hasil terbaru di `results/` (rata-rata 3 pengulangan per titik). "
         "Baseline dan optimized diukur pada sesi yang sama dengan protokol yang sama (`loadtest/run_full.sh`).", ""]

    for col, title in (("ugm", "KB resmi UGM (1.920 chunk)"), ("dummy_l", "Korpus sintetis 10⁶ chunk")):
        b, o = lt.get(("baseline", col, False)), lt.get(("optimized", col, False))
        if b and o:
            L += [f"## Retrieval saja — {title}", "", f"`{b['dir']}` vs `{o['dir']}`", ""] + retrieval_table(b, o)
            L += [""] + stage_table(b, o, 1) + [""]

    runs: list[tuple[str, dict, Path]] = []
    if a.llm_ref and (Path(a.llm_ref) / "summary.json").exists():
        ref = Path(a.llm_ref)
        runs.append(("Baseline asli (Ollama bawaan)", json.loads((ref / "summary.json").read_text()), ref))
    for mode, lbl in (("baseline", "Baseline (llama-server, serial)"), ("optimized", "Optimized")):
        d = lt.get((mode, "ugm", True))
        if d:
            runs.append((lbl, d, root / "loadtest" / d["dir"]))
    if len(runs) >= 2:
        L += ["## End-to-end dengan LLM — KB resmi UGM", "",
              "- *Baseline asli*: Ollama bawaan (25/34 layer di GPU, satu permintaan per waktu, parameter sampling "
              "bawaan model: `top_k 20`, `presence_penalty 1.5`).",
              "- *Baseline (llama-server, serial)*: pipeline baseline di atas runtime LLM teroptimasi — ablation untuk "
              "memisahkan efek runtime dari efek batching. Parameter sampling bawaan llama.cpp menghasilkan jawaban "
              "lebih panjang, jadi bandingkan dengan baseline asli memakai **throughput token**.",
              "- *Optimized*: pipeline teroptimasi (retrieval CUDA/OpenMP/FP16/batching + 4 permintaan LLM sekaligus) "
              "di runtime yang sama dengan baris tengah.", ""]
        L += e2e_tables(runs, root / "requests")

    builds = latest_builds(root, a.since)
    rows = []
    for src, lbl in (("raw", "14 dokumen resmi"), ("files_pdf_1000", "1.000 PDF sintetis")):
        b, o = builds.get(("baseline", src)), builds.get(("optimized", src))
        if not (b and o):
            continue
        sb, so = b["summary"], o["summary"]
        def ing(s: dict) -> float:  # waktu dinding ingest (format lama: jumlah tahap ingest_*)
            if "ingest_wall" in s:
                return s["ingest_wall"]["mean_ms"]
            return sum(v["mean_ms"] for k, v in s.items() if k.startswith("ingest_"))
        rows.append(f"| {lbl} | {sb['total']['mean_ms'] / 1000:.1f} s | {so['total']['mean_ms'] / 1000:.1f} s | "
                    f"{ratio(sb['total']['mean_ms'], so['total']['mean_ms'])} | "
                    f"{f0((ing(sb) or 0) / 1000, 1)} → {f0((ing(so) or 0) / 1000, 1)} s | "
                    f"{sb['index_embed']['mean_ms'] / 1000:.1f} → {so['index_embed']['mean_ms'] / 1000:.1f} s |")
    if rows:
        L += ["## Waktu build indeks dari nol", "",
              "| Masukan | Baseline | Optimized | Percepatan | Ingest (parse+clean+chunk) | Embedding |",
              "|---|---:|---:|---:|---:|---:|"] + rows + [""]

    probes = sorted((root / "llm_probe").glob("*.json"))
    if probes:
        L += ["## Server LLM (tools.llm_probe, prompt RAG ±2.000 token)", "",
              "| Konfigurasi | Decode 1 permintaan (tok/s) | Agregat 4 permintaan (tok/s) | Waktu 4 jawaban (s) |",
              "|---|---:|---:|---:|"]
        for f in probes:
            d = json.loads(f.read_text())
            r1, r4 = d["runs"].get("1", {}), d["runs"].get("4", {})
            L.append(f"| {d['label']} | {r1.get('decode_tok_s_per_request', '-')} | {r4.get('aggregate_decode_tok_s', '-')} | "
                     f"{f0((r4.get('wall_ms_per_round') or 0) / 1000, 1)} |")
        L.append("")

    abl = sorted((root / "ablation").glob("ablation-*.md"))
    if abl:
        L += [f"## Ablation per optimasi", "", f"Lihat `{abl[-1].relative_to(root.parent)}`.", ""]

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nTersimpan: {a.out}")


if __name__ == "__main__":
    main()
