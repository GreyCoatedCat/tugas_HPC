"""Orkestrator uji beban: sapu beberapa tingkat konkurensi & catat metrik (Tabel 3.2).

Untuk setiap jumlah pengguna (default 1, 10, 50, 100) dan setiap pengulangan:
  - jalankan Locust headless selama --duration detik,
  - sampling utilisasi CPU/GPU selama run,
  - kumpulkan latensi e2e p50/p95, TTFT p50/p95, throughput (QPS), latensi
    rata-rata per tahap pipeline, tingkat kegagalan.
Hasil diringkas sebagai rata-rata ± simpangan baku antar pengulangan.

    python -m loadtest.run --users 1 10 50 100 --duration 60 --repeats 3
    python -m loadtest.run --users 1 10 50 --no-generate --collection dummy_m   # retrieval saja
    python -m loadtest.run --users 5 --duration 30 --questions data/dummy/dummy_s/questions.txt --collection dummy_s

Output: results/loadtest/<timestamp>_<mode>_<koleksi>/{summary.md, summary.csv, summary.json, raw/...}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from rag.config import ROOT, load_config
from rag.metrics import ResourceSampler

LOCUSTFILE = Path(__file__).with_name("locustfile.py")


def parse_locust_csv(prefix: Path) -> dict:
    path = Path(f"{prefix}_stats.csv")
    out: dict = {"stages": {}}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            typ, name = row["Type"], row["Name"]

            def num(key):
                try:
                    return float(row[key])
                except (TypeError, ValueError, KeyError):
                    return None

            if typ == "METRIC" and name in ("e2e", "ttft", "abandoned"):
                out[name] = {
                    "count": int(num("Request Count") or 0),
                    "failures": int(num("Failure Count") or 0),
                    "p50": num("50%"),
                    "p95": num("95%"),
                    "p99": num("99%"),
                    "mean": num("Average Response Time"),
                    "max": num("Max Response Time"),
                }
            elif typ == "STAGE":
                out["stages"][name] = num("Average Response Time")
            elif typ == "POST" and name == "/api/chat":
                out["chat_failures"] = int(num("Failure Count") or 0)
    return out


def run_once(a, users: int, rep: int, raw_dir: Path) -> dict:
    prefix = raw_dir / f"u{users}_r{rep}"
    env = {
        **os.environ,
        "RAG_MODE": a.mode,
        "RAG_GENERATE": "1" if a.generate else "0",
        "RAG_WAIT_MIN": str(a.wait[0]),
        "RAG_WAIT_MAX": str(a.wait[1]),
        "RAG_ABANDONED_FILE": f"{prefix}_abandoned.txt",
    }
    if a.collection:
        env["RAG_COLLECTION"] = a.collection
    if a.questions:
        env["RAG_QUESTIONS"] = str(Path(a.questions).resolve())
    cmd = [
        sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
        "-u", str(users), "-r", str(a.spawn_rate or users), "--run-time", f"{a.duration}s",
        "--host", a.host, "--csv", str(prefix), "--only-summary", "--stop-timeout", str(a.stop_timeout),
        "--loglevel", "WARNING",
    ]
    sampler = ResourceSampler(interval_s=0.5)
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    sampler.stop()
    (raw_dir / f"u{users}_r{rep}.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    res = parse_locust_csv(prefix)
    ab_file = Path(f"{prefix}_abandoned.txt")
    abandoned = int(ab_file.read_text().strip() or 0) if ab_file.exists() else 0
    hist = list(sampler.history)
    e2e = res.get("e2e") or {}
    completed = e2e.get("count", 0) - e2e.get("failures", 0)
    gpu = [s["gpus"][0] for s in hist if s.get("gpus")]
    return {
        "users": users,
        "repeat": rep,
        "duration_s": a.duration,
        "wall_s": round(elapsed, 1),
        "completed": completed,
        "failures": e2e.get("failures", 0),
        "abandoned": abandoned,
        # dibagi waktu run sebenarnya (termasuk masa tunggu stop-timeout), bukan --duration:
        # request yang selesai saat antrean dikuras juga dihitung di `completed`.
        "qps": round(completed / elapsed, 4) if elapsed > 0 else None,
        "e2e_p50_ms": e2e.get("p50"),
        "e2e_p95_ms": e2e.get("p95"),
        "e2e_mean_ms": e2e.get("mean"),
        "ttft_p50_ms": (res.get("ttft") or {}).get("p50"),
        "ttft_p95_ms": (res.get("ttft") or {}).get("p95"),
        "stage_mean_ms": res.get("stages", {}),
        "cpu_percent_mean": round(statistics.fmean([s["cpu_percent"] for s in hist]), 1) if hist else None,
        "cpu_percent_max": max((s["cpu_percent"] for s in hist), default=None),
        "gpu_util_mean": round(statistics.fmean([g["util_percent"] for g in gpu]), 1) if gpu else None,
        "gpu_mem_max_mb": max((g["mem_used_mb"] for g in gpu), default=None),
        "ram_used_max_gb": max((s["ram_used_gb"] for s in hist), default=None),
        "locust_returncode": proc.returncode,
    }


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(mean, std, nd=0):
    if mean is None:
        return "-"
    return f"{mean:.{nd}f}" + (f" ± {std:.{nd}f}" if std else "")


def summarize(runs: list[dict], out_dir: Path, a) -> None:
    levels = sorted({r["users"] for r in runs})
    keys = ["qps", "e2e_p50_ms", "e2e_p95_ms", "ttft_p50_ms", "ttft_p95_ms", "cpu_percent_mean", "gpu_util_mean",
            "gpu_mem_max_mb", "ram_used_max_gb", "completed", "failures", "abandoned"]
    summary = []
    for u in levels:
        rs = [r for r in runs if r["users"] == u]
        row = {"users": u, "repeats": len(rs)}
        for k in keys:
            m, s = agg([r.get(k) for r in rs])
            row[k] = m
            row[k + "_std"] = s
        stages = sorted({st for r in rs for st in r["stage_mean_ms"]})
        row["stage_mean_ms"] = {st: agg([r["stage_mean_ms"].get(st) for r in rs])[0] for st in stages}
        summary.append(row)

    (out_dir / "summary.json").write_text(json.dumps({"config": vars(a), "runs": runs, "summary": summary},
                                                     indent=2, default=str), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        stage_names = sorted({st for row in summary for st in row["stage_mean_ms"]})
        w = csv.writer(f)
        w.writerow(["users"] + [x for k in keys for x in (k, k + "_std")] + [f"stage_{s}_ms" for s in stage_names])
        for row in summary:
            w.writerow([row["users"]] + [row.get(x) for k in keys for x in (k, k + "_std")]
                       + [row["stage_mean_ms"].get(s) for s in stage_names])

    lines = [
        f"# Hasil uji beban — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Mode: `{a.mode}` | Koleksi: `{a.collection or 'default'}` | Generasi LLM: {'ya' if a.generate else 'tidak (retrieval saja)'}",
        f"- Durasi per run: {a.duration}s | Pengulangan: {a.repeats} | Jeda pengguna: {a.wait[0]}–{a.wait[1]}s | Host: {a.host}",
        "",
        "| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['users']} | {fmt(r['completed'], r['completed_std'])} | {fmt(r['failures'], r['failures_std'])} | "
            f"{fmt(r['abandoned'], r['abandoned_std'])} | {fmt(r['qps'], r['qps_std'], 3)} | {fmt(r['e2e_p50_ms'], r['e2e_p50_ms_std'])} | "
            f"{fmt(r['e2e_p95_ms'], r['e2e_p95_ms_std'])} | {fmt(r['ttft_p50_ms'], r['ttft_p50_ms_std'])} | "
            f"{fmt(r['ttft_p95_ms'], r['ttft_p95_ms_std'])} | {fmt(r['cpu_percent_mean'], r['cpu_percent_mean_std'], 1)} | "
            f"{fmt(r['gpu_util_mean'], r['gpu_util_mean_std'], 1)} | {fmt(r['gpu_mem_max_mb'], None)} | {fmt(r['ram_used_max_gb'], None, 1)} |"
        )
    stage_names = sorted({st for row in summary for st in row["stage_mean_ms"]})
    if stage_names:
        lines += ["", "## Latensi rata-rata per tahap (ms, sisi server)", "",
                  "| Tahap | " + " | ".join(f"{r['users']} pengguna" for r in summary) + " |",
                  "|---|" + "---:|" * len(summary)]
        for st in stage_names:
            vals = [r["stage_mean_ms"].get(st) for r in summary]
            lines.append(f"| {st} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in vals) + " |")
    lines += ["", "Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. "
              "Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request "
              "yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan "
              f"(setelah stop-timeout {a.stop_timeout}s); server membatalkannya agar tidak membebani run berikutnya."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, nargs="+", default=[1, 10, 50, 100])
    ap.add_argument("--duration", type=int, default=60, help="detik per run")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--questions", default=None, help="file pertanyaan (default loadtest/questions_ugm.txt)")
    ap.add_argument("--no-generate", dest="generate", action="store_false", help="uji retrieval saja (tanpa LLM)")
    ap.add_argument("--host", default=f"http://{cfg['server']['host']}:{cfg['server']['port']}")
    ap.add_argument("--spawn-rate", type=float, default=None)
    ap.add_argument("--wait", type=float, nargs=2, default=[1.0, 3.0], metavar=("MIN", "MAX"))
    ap.add_argument("--stop-timeout", type=int, default=30, help="detik menunggu request berjalan selesai di akhir run")
    ap.add_argument("--cooldown", type=int, default=5, help="jeda antar run (detik)")
    ap.add_argument("--resummarize", metavar="DIR", help="hitung ulang ringkasan dari DIR/summary.json (tanpa menjalankan uji)")
    a = ap.parse_args(argv)

    if a.resummarize:
        out_dir = Path(a.resummarize)
        data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        runs = data["runs"]
        for r in runs:  # rumus throughput terbaru
            r["qps"] = round(r["completed"] / r["wall_s"], 4) if r.get("wall_s") else None
        saved = argparse.Namespace(**{**vars(a), **data["config"], "resummarize": None})
        summarize(runs, out_dir, saved)
        return

    try:
        h = httpx.get(f"{a.host}/api/health", timeout=10).json()
        print(f"Server OK — LLM: {h.get('llm_name')} ({'siap' if h['llm']['ok'] else 'TIDAK SIAP'})")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Server tidak bisa dihubungi di {a.host}: {exc}\nJalankan dulu: python -m server")

    tag = f"{datetime.now():%Y%m%d-%H%M%S}_{a.mode}_{a.collection or 'default'}{'' if a.generate else '_retrieval'}"
    out_dir = Path(cfg["paths"]["results_dir"]) / "loadtest" / tag
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for users in a.users:
        for rep in range(1, a.repeats + 1):
            print(f"\n=== {users} pengguna, run {rep}/{a.repeats} ({a.duration}s) ===")
            r = run_once(a, users, rep, raw_dir)
            runs.append(r)
            print(f"selesai={r['completed']} gagal={r['failures']} ditinggal={r['abandoned']} qps={r['qps']} e2e p50/p95={r['e2e_p50_ms']}/{r['e2e_p95_ms']} ms "
                  f"ttft p50={r['ttft_p50_ms']} ms cpu={r['cpu_percent_mean']}% gpu={r['gpu_util_mean']}%")
            time.sleep(a.cooldown)
    print()
    summarize(runs, out_dir, a)
    print(f"\nHasil lengkap: {out_dir}")


if __name__ == "__main__":
    main()
