"""Demo UI Gradio: chat + sumber + latensi per tahap, manajemen knowledge base,
monitor beban & uji beban langsung.

UI ini adalah klien dari API server (python -m server), sehingga pengguna UI
dan pengguna virtual uji beban melewati jalur yang sama.

Jalankan:  python -m ui
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from rag.config import ROOT, load_config

CFG = load_config()
API = os.environ.get("RAG_API_URL") or (CFG.get("ui") or {}).get("api_url", "http://127.0.0.1:8000")
CATEGORIES = ["peraturan", "kalender", "ukt", "beasiswa", "penerimaan", "kurikulum", "berita", "lainnya"]
STAGE_LABELS = {
    "queue_retrieval": "Antre retrieval",
    "embed_query": "Embedding kueri",
    "search_vector": "Pencarian vektor",
    "search_bm25": "Pencarian BM25",
    "fusion": "Fusi (RRF)",
    "fetch_chunks": "Ambil chunk",
    "rerank": "Reranking",
    "queue_llm": "Antre LLM",
    "llm_prefill": "LLM prefill (s.d. token pertama)",
    "llm_decode": "LLM decode",
}
EXAMPLES = [
    "Apa syarat mengajukan cuti akademik dan sampai kapan batas pengajuannya?",
    "Bagaimana prosedur pengajuan penurunan UKT?",
    "Berapa SKS maksimum yang boleh diambil jika IP semester sebelumnya di bawah 2,5?",
    "Kapan semester gasal 2026/2027 dimulai?",
    "Berapa lama masa studi maksimal program sarjana?",
]

client = httpx.Client(base_url=API, timeout=httpx.Timeout(600.0, connect=5.0))


def _kw(cls, **kw):
    """Buang argumen yang tidak didukung versi Gradio terpasang."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return cls(**kw)
    return cls(**{k: v for k, v in kw.items() if k in params})


def api_get(path: str, **params) -> Any:
    r = client.get(path, params=params or None)
    r.raise_for_status()
    return r.json()


def _err(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.json().get("detail") or exc.response.text
        except Exception:  # noqa: BLE001
            return exc.response.text
    if isinstance(exc, httpx.ConnectError):
        return f"API server tidak dapat dihubungi di {API}. Jalankan `python -m server`."
    return str(exc)


# ====================================================================== status
def status_md() -> str:
    try:
        h = api_get("/api/health")
        llm = h["llm"]
        llm_state = "✅ siap" if llm.get("ok") else f"⚠️ {llm.get('detail') or 'tidak siap'}"
        hpc = h.get("hpc") or {}
        if hpc.get("loaded"):
            gpu = hpc.get("gpu") or {}
            hpc_state = (f"✅ `rag_hpc {hpc.get('version', '')}` · {gpu.get('name', 'GPU')} (sm {gpu.get('compute_capability', '?')})"
                         f" · OpenMP {hpc.get('omp_max_threads', '?')} thread")
        else:
            hpc_state = f"⚠️ {hpc.get('detail') or 'modul C++ belum dibangun'}"
        return (
            f"**API** ✅ `{API}` · **LLM** `{h['llm_name']}` {llm_state} · **Embedder** `{h['embedder']}`"
            f" · **Reranker** `{h['reranker'] or 'nonaktif'}`\n\n**Akselerasi C++** {hpc_state}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"**API** ❌ {_err(exc)}"


def mode_choices() -> tuple[list[tuple[str, str]], dict]:
    try:
        m = api_get("/api/modes")["modes"]
    except Exception:  # noqa: BLE001
        return [("baseline", "baseline")], {}
    return [(n if s["available"] else f"{n} (belum tersedia)", n) for n, s in m.items()], m


def collection_choices() -> list[tuple[str, str]]:
    try:
        cols = api_get("/api/collections")
    except Exception:  # noqa: BLE001
        return [("ugm", "ugm")]
    out = []
    for c in cols:
        if c["name"] == "bench-build-tmp":
            continue
        label = f"{c['name']} — {c['n_documents']} dok, {c['n_chunks']} chunk"
        if c["synthetic"]:
            label = f"[SINTETIS] {label}"
        out.append((label, c["name"]))
    return out or [("ugm", "ugm")]


def refresh_header():
    choices, _ = mode_choices()
    cols = collection_choices()
    return status_md(), gr.update(choices=choices), gr.update(choices=cols)


def on_mode_change(mode: str):
    _, info = mode_choices()
    st = info.get(mode)
    if st and not st["available"]:
        gr.Warning(f"Mode '{mode}' belum tersedia: {', '.join(st['missing'])}. Kembali ke baseline.")
        return gr.update(value="baseline"), st.get("description", "")
    return gr.update(), (st or {}).get("description", "")


# ====================================================================== chat
def render_sources(sources: list[dict], cited: list[int] | None = None, synthetic: bool = False) -> str:
    if not sources:
        return "_Tidak ada potongan dokumen yang cukup relevan di knowledge base._"
    lines = []
    if synthetic:
        lines.append("> ⚠️ **Koleksi SINTETIS** — data dummy untuk pengujian, bukan informasi resmi.\n")
    for s in sources:
        page = ""
        if s.get("page_start"):
            page = f" · hlm. {s['page_start']}" + (f"–{s['page_end']}" if s.get("page_end") not in (None, s["page_start"]) else "")
        mark = " ✅ dikutip" if cited and s["n"] in cited else ""
        link = f" · [buka sumber]({s['url']})" if s.get("url") else ""
        score = f" · relevansi {s['rerank_score']:.2f}" if s.get("rerank_score") is not None else ""
        cat = f" · {s['category']}" if s.get("category") else ""
        text = s["text"].replace("\n", "\n\n")
        lines.append(
            f"**[{s['n']}] {s['title']}**{page}{cat}{score}{link}{mark}\n\n"
            f"<details><summary>Lihat kutipan</summary>\n\n{text}\n\n</details>\n"
        )
    return "\n".join(lines)


def render_batch(batch: dict | None) -> str:
    if not batch:
        return ""
    parts = [f"⚡ diproses dalam **batch {batch.get('batch_size', 1)} permintaan**"]
    if (batch.get("overlap_saved_ms") or 0) >= 0.1:
        parts.append(f"BM25 (CPU) ∥ vektor (GPU) menghemat {batch['overlap_saved_ms']:.1f} ms")
    k = batch.get("kernel_ms") or {}
    if "vec_score_ms" in k:
        parts.append(f"kernel CUDA similarity {k['vec_score_ms']:.2f} ms + top-k {k.get('vec_select_ms', 0):.2f} ms")
    if "bm25_parallel_ms" in k:
        parts.append(f"BM25 OpenMP {k['bm25_parallel_ms']:.2f} ms")
    if batch.get("rerank_pairs"):
        parts.append(f"rerank {batch['rerank_pairs']} pasangan FP16")
    return " · ".join(parts)


def render_timings(timings: dict[str, float], ttft: float | None = None, e2e: float | None = None,
                   llm: dict | None = None, batch: dict | None = None) -> str:
    if not timings:
        return ""
    total = e2e or sum(timings.values()) or 1.0
    rows = []
    for k, v in timings.items():
        pct = 100.0 * v / total
        filled = min(10, int(round(pct / 10)))
        bar = "█" * filled + "░" * (10 - filled)
        rows.append(f"- `{bar}` {pct:5.1f}% · **{STAGE_LABELS.get(k, k)}** · {v:,.1f} ms")
    head = []
    if ttft is not None:
        head.append(f"**TTFT** {ttft:,.0f} ms")
    if e2e is not None:
        head.append(f"**Total** {e2e:,.0f} ms")
    if llm and llm.get("completion_tokens") and llm.get("backend_decode_ms"):
        tps = llm["completion_tokens"] / (llm["backend_decode_ms"] / 1000)
        head.append(f"**Decode** {tps:.1f} token/s ({llm['completion_tokens']} token)")
    extra = render_batch(batch)
    return (" · ".join(head) + "\n\n" if head else "") + "\n".join(rows) + (f"\n\n{extra}" if extra else "")


def _msg_text(content) -> str:
    """Isi pesan Chatbot bisa berupa string atau daftar blok (format Gradio 6)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content or "")


def user_submit(message: str, history: list):
    """Masukkan pesan pengguna & kunci input sampai jawaban selesai."""
    message = (message or "").strip()
    history = list(history or [])
    if message:
        history.append({"role": "user", "content": message})
    return gr.update(value="", interactive=False), history, gr.update(interactive=False)


def unlock_input():
    return gr.update(interactive=True), gr.update(interactive=True)


def bot_respond(history: list, mode: str, collection: str):
    history = list(history or [])
    if not history or history[-1].get("role") != "user":
        yield history, gr.update(), gr.update()
        return
    message = _msg_text(history[-1]["content"]).strip()
    history.append({"role": "assistant", "content": "⏳ _Mencari di knowledge base..._"})
    yield history, "_Mencari..._", ""
    payload = {"question": message, "collection": collection, "mode": mode, "stream": True, "client": "ui"}
    answer, sources, synthetic, batch = "", [], False, {}
    last = 0.0
    try:
        with client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                r.read()
                raise httpx.HTTPStatusError("error", request=r.request, response=r)
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "meta":
                    sources, synthetic, batch = ev["sources"], ev.get("synthetic", False), ev.get("batch") or {}
                    history[-1]["content"] = "✍️ _Menyusun jawaban dari dokumen..._" if sources else ""
                    yield (history, render_sources(sources, synthetic=synthetic),
                           render_timings(ev.get("timings_ms", {}), batch=batch))
                elif ev["type"] == "token":
                    answer += ev["text"]
                    now = time.perf_counter()
                    if now - last > 0.05:
                        history[-1]["content"] = answer
                        last = now
                        yield history, gr.update(), gr.update()
                elif ev["type"] == "done":
                    final = ev["answer"] or answer
                    if ev.get("uncited_warning"):
                        final += "\n\n> ⚠️ Jawaban tidak menyertakan nomor sumber — mohon verifikasi dengan dokumen di panel Sumber."
                    if synthetic:
                        final = "> ⚠️ Dijawab dari koleksi **SINTETIS** (data dummy).\n\n" + final
                    history[-1]["content"] = final
                    shown = [] if ev.get("refused") else sources
                    yield (history, render_sources(shown, ev.get("cited"), synthetic),
                           render_timings(ev["timings_ms"], ev.get("ttft_ms"), ev.get("e2e_ms"), ev.get("llm"), batch))
                elif ev["type"] == "error":
                    history[-1]["content"] = f"⚠️ {ev['message']}"
                    yield history, gr.update(), gr.update()
    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = f"⚠️ {_err(exc)}"
        yield history, gr.update(), gr.update()


def compare_fn(message: str, history: list, collection: str):
    """Jalankan retrieval pertanyaan yang sama di mode baseline & teroptimasi, tampilkan percepatan per tahap."""
    q = (message or "").strip()
    if not q:
        users = [_msg_text(m["content"]) for m in (history or []) if m.get("role") == "user"]
        q = users[-1].strip() if users else ""
    if not q:
        return "_Tulis pertanyaan di kotak chat (atau kirim satu pertanyaan) lalu tekan **Bandingkan**._"
    _, info = mode_choices()
    modes = [m for m in ("baseline", "optimized") if (info.get(m) or {}).get("available", m == "baseline")]
    if len(modes) < 2:
        return "⚠️ Mode teroptimasi belum tersedia — bangun modul C++ dulu: `bash cpp/build.sh`."
    reps = 5
    med: dict[str, dict[str, float]] = {}
    top: dict[str, list[int]] = {}
    try:
        for m in modes:
            payload = {"question": q, "collection": collection, "mode": m, "generate": False, "stream": False,
                       "client": "ui-compare"}
            client.post("/api/chat", json=payload)  # pemanasan (muat snapshot/backend)
            runs = [client.post("/api/chat", json=payload).json() for _ in range(reps)]
            stages = sorted({k for r in runs for k in (r.get("timings_ms") or {}) if not k.startswith("queue")})
            med[m] = {k: sorted(r["timings_ms"].get(k, 0.0) for r in runs)[reps // 2] for k in stages}
            med[m]["__total"] = sorted(sum(v for k, v in r["timings_ms"].items() if not k.startswith("queue"))
                                       for r in runs)[reps // 2]
            top[m] = [s_["chunk_id"] for s_ in runs[-1].get("sources", [])]
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}"
    b, o = med["baseline"], med["optimized"]
    rows = ["| Tahap | Baseline | Teroptimasi | Percepatan |", "|---|---:|---:|---:|"]
    for k in [k for k in b if k != "__total"]:
        sp = b[k] / o[k] if o.get(k) else None
        rows.append(f"| {STAGE_LABELS.get(k, k)} | {b[k]:,.1f} ms | {o.get(k, 0):,.1f} ms | "
                    f"{'-' if sp is None else f'{sp:,.1f}×'} |")
    sp_total = b["__total"] / o["__total"] if o["__total"] else 0
    rows.append(f"| **Total retrieval** | **{b['__total']:,.1f} ms** | **{o['__total']:,.1f} ms** | **{sp_total:,.1f}×** |")
    same = "✅ identik" if top["baseline"] == top["optimized"] else "⚠️ berbeda (cek urutan skor seri)"
    return (f"**Pertanyaan:** {q}\n\nMedian {reps} kali, koleksi `{collection}`, tanpa antrean & tanpa LLM.\n\n"
            + "\n".join(rows)
            + f"\n\nSumber konteks yang dipilih: {same}.\n\n<sub>Generasi LLM memakai model & server yang sama di kedua mode; "
            "keunggulannya (batching beberapa permintaan sekaligus) terlihat saat uji beban.</sub>")


# ====================================================================== knowledge base
def kb_view(collection: str):
    try:
        d = api_get(f"/api/kb/{collection}/documents")
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}", []
    st = d["stats"]
    idx = st.get("index") or {}
    t = idx.get("timings_ms") or {}
    info = [
        f"### Koleksi `{st['name']}`" + (" — **SINTETIS**" if st["synthetic"] else ""),
        f"{st['description']}",
        f"- Dokumen: **{st['n_documents']}** · Chunk: **{st['n_chunks']}**",
        f"- Indeks: {idx.get('n_chunks', 0)} chunk · model `{idx.get('embed_model', '-')}` · dibangun {idx.get('built_at', '-')}"
        + (" · ⚠️ **perlu dibangun ulang**" if st["index_stale"] else ""),
    ]
    if t:
        info.append("- Waktu build terakhir: " + ", ".join(f"{k} {v / 1000:.2f}s" for k, v in t.items()))
    rows = [
        [doc["id"], doc["title"], doc["category"] or "", doc["n_chunks"], doc["source_type"] or "",
         doc["source_url"] or "", (doc["added_at"] or "")[:19]]
        for doc in d["documents"]
    ]
    return "\n".join(info), rows


def _fmt_index(index: dict | None) -> str:
    if not index:
        return ""
    t = index.get("timings_ms") or {}
    return (f"\n\n**Indeks diperbarui:** {index['n_chunks']} chunk ({index.get('n_embedded', 0)} di-embed baru) — "
            + ", ".join(f"{k} {v:,.0f} ms" for k, v in t.items()))


def upload_fn(collection: str, files, title: str, category: str, source_url: str):
    if not files:
        return "⚠️ Pilih minimal satu file.", *kb_view(collection)
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files if isinstance(files, list) else [files])]
    handles = [("files", (Path(p).name, open(p, "rb"))) for p in paths]
    try:
        r = client.post(
            f"/api/kb/{collection}/documents", files=handles,
            data={"title": title or "", "category": category or "", "source_url": source_url or "", "rebuild": "true"},
        )
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}", *kb_view(collection)
    finally:
        for _, (_, fh) in handles:
            fh.close()
    lines = []
    for x in res["results"]:
        name = x.get("title") or x.get("filename")
        if x["status"] == "ok":
            lines.append(f"- ✅ **{name}** — {x['n_chunks']} chunk, {x['n_chars']:,} karakter")
        else:
            lines.append(f"- ❌ **{name}** — {x.get('reason') or x.get('error')}")
        for w in x.get("warnings") or []:
            lines.append(f"  - ⚠️ {w}")
    return "\n".join(lines) + _fmt_index(res.get("index")), *kb_view(collection)


def url_fn(collection: str, url: str, title: str, category: str):
    if not (url or "").startswith(("http://", "https://")):
        return "⚠️ Masukkan URL lengkap (http/https).", *kb_view(collection)
    try:
        r = client.post(f"/api/kb/{collection}/url", json={"url": url, "title": title or None, "category": category or None})
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}", *kb_view(collection)
    x = res["result"]
    msg = (f"✅ **{x.get('title')}** — {x.get('n_chunks')} chunk" if x["status"] == "ok"
           else f"❌ {x.get('reason') or x.get('error')}")
    for w in x.get("warnings") or []:
        msg += f"\n- ⚠️ {w}"
    return msg + _fmt_index(res.get("index")), *kb_view(collection)


def chunks_fn(collection: str, doc_id):
    if not doc_id:
        return "Masukkan ID dokumen."
    try:
        d = api_get(f"/api/kb/{collection}/documents/{int(doc_id)}/chunks", limit=30)
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}"
    doc = d["document"]
    out = [f"### {doc['title']}", f"Sumber: {doc.get('source_url') or doc.get('file_path') or '-'}", ""]
    for c in d["chunks"]:
        pg = f" (hlm. {c['page_start']}–{c['page_end']})" if c.get("page_start") else ""
        out.append(f"**Chunk #{c['ordinal']}**{pg} · {c['n_words']} kata\n\n```text\n{c['text']}\n```")
    if doc["n_chunks"] > len(d["chunks"]):
        out.append(f"_… {doc['n_chunks'] - len(d['chunks'])} chunk lainnya tidak ditampilkan._")
    return "\n\n".join(out)


def delete_fn(collection: str, doc_id, confirm: bool):
    if not confirm:
        return "⚠️ Centang konfirmasi dulu untuk menghapus.", *kb_view(collection), False
    try:
        r = client.delete(f"/api/kb/{collection}/documents/{int(doc_id)}")
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}", *kb_view(collection), False
    return f"🗑️ Dokumen {int(doc_id)} dihapus ({res['deleted_chunks']} chunk)." + _fmt_index(res.get("index")), *kb_view(collection), False


def rebuild_fn(collection: str, full: bool):
    try:
        r = client.post(f"/api/kb/{collection}/rebuild", json={"full": full, "mode": "baseline"})
        r.raise_for_status()
        meta = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}", *kb_view(collection)
    return ("✅ Rebuild " + ("penuh" if full else "inkremental") + _fmt_index(meta)), *kb_view(collection)


def create_collection_fn(name: str, desc: str, synthetic: bool):
    try:
        r = client.post("/api/collections", json={"name": (name or "").strip(), "description": desc or "", "synthetic": synthetic})
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}", gr.update()
    return f"✅ Koleksi `{name}` dibuat.", gr.update(choices=collection_choices(), value=name.strip())


# ====================================================================== monitor & uji beban
LOADTEST: dict[str, Any] = {"proc": None, "started": None, "cfg": None, "prefix": None}


def metrics_md() -> str:
    try:
        m = api_get("/api/metrics", window=60)
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {_err(exc)}"
    res = m.get("resources") or {}
    gpus = res.get("gpus") or []
    g = gpus[0] if gpus else {}
    fmt = lambda v, u="": "-" if v is None else f"{v:,.0f}{u}"  # noqa: E731
    lines = [
        f"**Sumber daya:** CPU {fmt(res.get('cpu_percent'), '%')} · GPU {fmt(g.get('util_percent'), '%')} · "
        f"VRAM {fmt(g.get('mem_used_mb'))}/{fmt(g.get('mem_total_mb'))} MB · RAM {res.get('ram_used_gb', '-')} GB",
        "",
        "| Mode | Sedang diproses | Selesai (60 dtk) | Throughput | E2E p50 | E2E p95 | TTFT p50 | TTFT p95 | Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, x in (m.get("by_mode") or {"semua": m}).items():
        lines.append(
            f"| **{name}** | {x['in_flight']} | {x['completed']} | {x['qps']:.2f} QPS | {fmt(x['e2e_p50_ms'], ' ms')} | "
            f"{fmt(x['e2e_p95_ms'], ' ms')} | {fmt(x['ttft_p50_ms'], ' ms')} | {fmt(x['ttft_p95_ms'], ' ms')} | {x['errors']} |"
        )
    for name, x in (m.get("by_mode") or {}).items():
        if x.get("stage_mean_ms"):
            lines += ["", f"**{name}** — rata-rata per tahap: "
                      + " · ".join(f"{STAGE_LABELS.get(k, k)} {v:,.0f} ms" for k, v in x["stage_mean_ms"].items())]
    return "\n".join(lines)


def loadtest_status() -> str:
    p = LOADTEST["proc"]
    if p is None:
        return "Tidak ada uji beban yang berjalan."
    cfg = LOADTEST["cfg"]
    elapsed = time.time() - LOADTEST["started"]
    if p.poll() is None:
        return f"🟢 **Berjalan** — {cfg['users']} pengguna virtual, {elapsed:.0f}/{cfg['duration']} detik"
    from loadtest.run import parse_locust_csv

    res = parse_locust_csv(LOADTEST["prefix"])
    e2e, ttft = res.get("e2e") or {}, res.get("ttft") or {}
    done = e2e.get("count", 0) - e2e.get("failures", 0)
    return (f"⚪ **Selesai** ({cfg['users']} pengguna, {cfg['duration']} detik) — selesai {done}, gagal {e2e.get('failures', 0)}, "
            f"throughput {done / max(1, cfg['duration']):.2f} QPS, E2E p50/p95 {e2e.get('p50')}/{e2e.get('p95')} ms, "
            f"TTFT p50/p95 {ttft.get('p50')}/{ttft.get('p95')} ms. CSV: `{LOADTEST['prefix']}_stats.csv`")


def start_loadtest(users, duration, generate: bool, mode: str, collection: str, questions: str):
    p = LOADTEST["proc"]
    if p is not None and p.poll() is None:
        return "⚠️ Uji beban masih berjalan."
    out_dir = Path(CFG["paths"]["results_dir"]) / "loadtest" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_u{int(users)}"
    env = {**os.environ, "RAG_MODE": mode, "RAG_COLLECTION": collection, "RAG_GENERATE": "1" if generate else "0"}
    if questions and questions.strip():
        env["RAG_QUESTIONS"] = str((ROOT / questions.strip()).resolve()) if not Path(questions.strip()).is_absolute() else questions.strip()
    cmd = [sys.executable, "-m", "locust", "-f", str(ROOT / "loadtest" / "locustfile.py"), "--headless",
           "-u", str(int(users)), "-r", str(max(1, int(users) // 5)), "--run-time", f"{int(duration)}s",
           "--host", API, "--csv", str(prefix), "--only-summary", "--stop-timeout", "20", "--loglevel", "WARNING"]
    LOADTEST.update(proc=subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                    started=time.time(), cfg={"users": int(users), "duration": int(duration)}, prefix=prefix)
    return loadtest_status()


def stop_loadtest():
    p = LOADTEST["proc"]
    if p is not None and p.poll() is None:
        p.terminate()
        return "🛑 Uji beban dihentikan."
    return loadtest_status()


# ====================================================================== layout
CSS = """
#title h1 {margin-bottom: 0}
.sources {max-height: 520px; overflow-y: auto}
footer {display: none !important}
"""


def build_ui() -> gr.Blocks:
    modes, _ = mode_choices()
    cols = collection_choices()
    default_col = (CFG.get("kb") or {}).get("default_collection", "ugm")
    with gr.Blocks(title="Asisten Informasi Akademik UGM") as demo:
        gr.Markdown("# 🎓 Asisten Informasi Akademik UGM\nChatbot RAG yang menjawab **hanya** dari dokumen resmi di knowledge base.",
                    elem_id="title")
        with gr.Row():
            status = gr.Markdown(status_md())
        with gr.Row():
            mode = gr.Radio(choices=modes, value="baseline", label="Mode pipeline", scale=1)
            collection = gr.Dropdown(choices=cols, value=default_col if any(v == default_col for _, v in cols) else cols[0][1],
                                     label="Koleksi knowledge base", scale=2)
            refresh = gr.Button("🔄 Muat ulang", scale=0)
        mode_desc = gr.Markdown("")

        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = _kw(gr.Chatbot, height=540, label="Percakapan", buttons=["copy", "copy_all"])
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Tulis pertanyaan tentang informasi akademik UGM...", show_label=False, scale=5)
                        send = gr.Button("Kirim", variant="primary", scale=1)
                    gr.Examples(EXAMPLES, inputs=msg, label="Contoh pertanyaan")
                    with gr.Row():
                        clear = gr.Button("🧹 Bersihkan percakapan", size="sm")
                        compare = gr.Button("⚖️ Bandingkan baseline vs teroptimasi", size="sm")
                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Sumber")
                    sources = gr.Markdown("_Sumber jawaban akan muncul di sini._", elem_classes=["sources"])
                    gr.Markdown("### ⏱️ Latensi per tahap")
                    timings = gr.Markdown("")
            with gr.Accordion("⚖️ Perbandingan baseline vs teroptimasi", open=True):
                comparison = gr.Markdown("_Tekan **Bandingkan** untuk mengukur pertanyaan yang sama di kedua mode._")
            gr.Markdown("<sub>Setiap pertanyaan dijawab secara independen. Jika dokumen yang relevan tidak ditemukan, "
                        "chatbot akan menolak menjawab alih-alih menebak.</sub>")

        with gr.Tab("📚 Knowledge Base"):
            kb_info = gr.Markdown("")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("➕ Tambah dokumen dari file", open=True):
                        up_files = gr.File(file_count="multiple", file_types=[".pdf", ".html", ".htm", ".txt", ".md"],
                                           label="File (PDF/HTML/TXT/MD)")
                        up_title = gr.Textbox(label="Judul (opsional, hanya jika 1 file)")
                        up_cat = gr.Dropdown(CATEGORIES, label="Kategori", allow_custom_value=True)
                        up_url = gr.Textbox(label="URL sumber resmi (opsional, untuk sitasi)")
                        up_btn = gr.Button("Unggah & indeks", variant="primary")
                    with gr.Accordion("🌐 Tambah dari URL resmi", open=False):
                        url_in = gr.Textbox(label="URL (halaman HTML atau PDF)")
                        url_title = gr.Textbox(label="Judul (opsional)")
                        url_cat = gr.Dropdown(CATEGORIES, label="Kategori", allow_custom_value=True)
                        url_btn = gr.Button("Ambil & indeks", variant="primary")
                    with gr.Accordion("🗂️ Buat koleksi baru", open=False):
                        new_name = gr.Textbox(label="Nama (huruf kecil, angka, - _)")
                        new_desc = gr.Textbox(label="Deskripsi")
                        new_syn = gr.Checkbox(label="Koleksi sintetis (data dummy untuk pengujian)")
                        new_btn = gr.Button("Buat koleksi")
                with gr.Column():
                    kb_msg = gr.Markdown("")
                    with gr.Row():
                        rb_inc = gr.Button("♻️ Rebuild inkremental")
                        rb_full = gr.Button("🔁 Rebuild penuh (ukur waktu build)")
                    with gr.Row():
                        doc_id = gr.Number(label="ID dokumen", precision=0)
                        view_btn = gr.Button("👁️ Lihat chunk")
                    with gr.Row():
                        del_confirm = gr.Checkbox(label="Saya yakin ingin menghapus dokumen ini")
                        del_btn = gr.Button("🗑️ Hapus dokumen", variant="stop")
            docs = _kw(gr.Dataframe, headers=["ID", "Judul", "Kategori", "Chunk", "Tipe", "URL sumber", "Ditambahkan"],
                       interactive=False, wrap=True, max_height=500, label="Dokumen di koleksi")
            chunk_view = gr.Markdown("")

        with gr.Tab("📈 Monitor & Uji Beban"):
            gr.Markdown("Metrik server diperbarui setiap 2 detik — gunakan chat di tab lain selagi uji beban berjalan.")
            metrics = gr.Markdown(metrics_md())
            with gr.Row():
                lt_users = gr.Number(value=10, label="Pengguna virtual", precision=0)
                lt_dur = gr.Number(value=60, label="Durasi (detik)", precision=0)
                lt_gen = gr.Checkbox(value=True, label="Dengan generasi LLM")
                lt_q = gr.Textbox(value="loadtest/questions_ugm.txt", label="File pertanyaan")
            with gr.Row():
                lt_start = gr.Button("▶️ Mulai uji beban", variant="primary")
                lt_stop = gr.Button("⏹️ Hentikan", variant="stop")
            lt_status = gr.Markdown(loadtest_status())
            timer = gr.Timer(2.0)
            timer.tick(lambda: (metrics_md(), loadtest_status()), outputs=[metrics, lt_status])

        # ---- event wiring
        refresh.click(refresh_header, outputs=[status, mode, collection])
        mode.change(on_mode_change, inputs=mode, outputs=[mode, mode_desc])
        for trigger in (send.click, msg.submit):
            trigger(user_submit, inputs=[msg, chatbot], outputs=[msg, chatbot, send], queue=False).then(
                bot_respond, inputs=[chatbot, mode, collection], outputs=[chatbot, sources, timings]
            ).then(unlock_input, outputs=[msg, send], queue=False)
        clear.click(lambda: ([], "_Sumber jawaban akan muncul di sini._", ""), outputs=[chatbot, sources, timings])
        compare.click(lambda: "⏳ _Mengukur kedua mode..._", outputs=comparison, queue=False).then(
            compare_fn, inputs=[msg, chatbot, collection], outputs=comparison)

        collection.change(kb_view, inputs=collection, outputs=[kb_info, docs])
        up_btn.click(upload_fn, inputs=[collection, up_files, up_title, up_cat, up_url], outputs=[kb_msg, kb_info, docs])
        url_btn.click(url_fn, inputs=[collection, url_in, url_title, url_cat], outputs=[kb_msg, kb_info, docs])
        view_btn.click(chunks_fn, inputs=[collection, doc_id], outputs=chunk_view)
        del_btn.click(delete_fn, inputs=[collection, doc_id, del_confirm], outputs=[kb_msg, kb_info, docs, del_confirm])
        rb_inc.click(lambda c: rebuild_fn(c, False), inputs=collection, outputs=[kb_msg, kb_info, docs])
        rb_full.click(lambda c: rebuild_fn(c, True), inputs=collection, outputs=[kb_msg, kb_info, docs])
        new_btn.click(create_collection_fn, inputs=[new_name, new_desc, new_syn], outputs=[kb_msg, collection])
        lt_start.click(start_loadtest, inputs=[lt_users, lt_dur, lt_gen, mode, collection, lt_q], outputs=lt_status)
        lt_stop.click(stop_loadtest, outputs=lt_status)
        demo.load(kb_view, inputs=collection, outputs=[kb_info, docs])
    return demo
