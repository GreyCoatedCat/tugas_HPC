"""Demo UI Gradio: chat + sumber + latensi per tahap, manajemen knowledge base,
monitor beban & uji beban langsung.

UI ini adalah klien dari API server (python -m server), sehingga pengguna UI
dan pengguna virtual uji beban melewati jalur yang sama.

Jalankan:  python -m ui
"""

from __future__ import annotations

import base64
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

from . import results as bench

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
def status_html() -> str:
    """Status sistem ringkas untuk kaki sidebar."""
    def row(ok: bool | None, label: str, detail: str) -> str:
        dot = "ok" if ok else ("warn" if ok is None else "bad")
        return f'<div class="st-row"><span class="dot {dot}"></span><b>{label}</b><span class="st-detail">{detail}</span></div>'

    try:
        h = api_get("/api/health")
    except Exception as exc:  # noqa: BLE001
        return row(False, "API", _err(exc))
    llm = h["llm"]
    hpc = h.get("hpc") or {}
    gpu = (hpc.get("gpu") or {}).get("name", "GPU").replace("NVIDIA GeForce ", "")
    rows = [
        row(True, "API", API.replace("http://", "")),
        row(bool(llm.get("ok")), "LLM", h["llm_name"].split(":", 1)[-1] if llm.get("ok") else (llm.get("detail") or "tidak siap")),
        row(bool(hpc.get("loaded")), "C++",
            f"{gpu} · OpenMP {hpc.get('omp_max_threads', '?')}" if hpc.get("loaded") else "rag_hpc belum dibangun"),
    ]
    return "".join(rows)


def mode_choices() -> tuple[list[tuple[str, str]], dict]:
    try:
        m = api_get("/api/modes")["modes"]
    except Exception:  # noqa: BLE001
        return [("baseline", "baseline")], {}
    nice = {"baseline": "Baseline", "optimized": "Optimized"}
    return [(nice.get(n, n) if s["available"] else f"{nice.get(n, n)} (belum tersedia)", n) for n, s in m.items()], m


def collection_choices() -> list[tuple[str, str]]:
    try:
        cols = api_get("/api/collections")
    except Exception:  # noqa: BLE001
        return [("ugm", "ugm")]
    out = []
    for c in cols:
        if c["name"] == "bench-build-tmp":
            continue
        label = f"{c['name']} · {c['n_chunks']:,} chunk".replace(",", ".")
        if c["synthetic"]:
            label = f"{label} · sintetis"
        out.append((label, c["name"]))
    return out or [("ugm", "ugm")]


def refresh_header():
    choices, _ = mode_choices()
    cols = collection_choices()
    return status_html(), gr.update(choices=choices), gr.update(choices=cols)


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
        lines.append("> **Koleksi sintetis** — data dummy untuk pengujian, bukan informasi resmi.\n")
    for s in sources:
        page = ""
        if s.get("page_start"):
            page = f" · hlm. {s['page_start']}" + (f"–{s['page_end']}" if s.get("page_end") not in (None, s["page_start"]) else "")
        mark = " · **dikutip**" if cited and s["n"] in cited else ""
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
    parts = [f"Diproses dalam **batch {batch.get('batch_size', 1)} permintaan**"]
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


def greeting_html() -> str:
    h = time.localtime().tm_hour
    salam = ("Selamat pagi" if 4 <= h < 11 else "Selamat siang" if h < 15 else "Selamat sore" if h < 18
             else "Selamat malam")
    return ('<div class="hero">'
            f'<div class="hero-card"><img src="{LOGO_STACKED}" alt="Universitas Gadjah Mada"></div>'
            f'<div class="hero-copy"><div class="hero-eyebrow">{salam}</div>'
            '<div class="hero-text">Ada yang ingin ditanyakan seputar akademik UGM?</div>'
            '<div class="hero-sub">Jawaban hanya bersumber dari dokumen akademik resmi UGM di knowledge base.</div>'
            '</div></div>')


def user_submit(message: str, history: list, recents: list):
    """Masukkan pesan pengguna, pindah ke tampilan percakapan, kunci input sampai jawaban selesai."""
    message = (message or "").strip()
    history = list(history or [])
    recents = list(recents or [])
    if message:
        history.append({"role": "user", "content": message})
        recents = ([message] + [r for r in recents if r != message])[:12]
    started = bool(history)
    return (
        gr.update(value="", interactive=False),          # msg
        gr.update(value=history, visible=started),         # chatbot
        gr.update(interactive=False),                      # send
        gr.update(visible=not started),                    # hero
        gr.update(visible=not started),                    # contoh
        gr.update(visible=started),                        # panel kanan
        recents,
        gr.update(choices=recents, value=None),            # daftar "Terbaru"
        gr.update(visible=not recents),                    # petunjuk "Terbaru" kosong
    )


def unlock_input():
    return gr.update(interactive=True), gr.update(interactive=True)


def new_chat():
    return (
        gr.update(value=[], visible=False), gr.update(value=greeting_html(), visible=True), gr.update(visible=True),
        gr.update(visible=False), "_Sumber jawaban akan muncul di sini._", "", COMPARE_HINT,
        gr.update(value=""), gr.Tabs(selected="chat"), *nav_state("chat"),
    )


NAV_TABS = ("chat", "kb", "mon", "hasil")


def nav_state(active: str):
    return [gr.update(variant="primary" if t == active else "secondary") for t in NAV_TABS]


def go(tab: str):
    return (gr.Tabs(selected=tab), *nav_state(tab))


def pick_recent(question: str | None):
    if not question:
        return gr.update(), gr.update(), *[gr.update()] * len(NAV_TABS)
    return gr.update(value=question), gr.Tabs(selected="chat"), *nav_state("chat")


def bot_respond(history: list, mode: str, collection: str):
    history = list(history or [])
    if not history or history[-1].get("role") != "user":
        yield history, gr.update(), gr.update()
        return
    message = _msg_text(history[-1]["content"]).strip()
    history.append({"role": "assistant", "content": "_Mencari di knowledge base…_"})
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
                    history[-1]["content"] = "_Menyusun jawaban dari dokumen…_" if sources else ""
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
                        final += "\n\n> **Perhatian:** jawaban tidak menyertakan nomor sumber — mohon verifikasi dengan dokumen di panel Sumber."
                    if synthetic:
                        final = "> Dijawab dari koleksi **sintetis** (data dummy).\n\n" + final
                    history[-1]["content"] = final
                    shown = [] if ev.get("refused") else sources
                    yield (history, render_sources(shown, ev.get("cited"), synthetic),
                           render_timings(ev["timings_ms"], ev.get("ttft_ms"), ev.get("e2e_ms"), ev.get("llm"), batch))
                elif ev["type"] == "error":
                    history[-1]["content"] = f"**Gagal:** {ev['message']}"
                    yield history, gr.update(), gr.update()
    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = f"**Gagal:** {_err(exc)}"
        yield history, gr.update(), gr.update()


def _compare_call(payload: dict) -> dict:
    r = client.post("/api/chat", json=payload)
    data = r.json()
    if r.status_code >= 400 or "timings_ms" not in data:
        raise RuntimeError(f"[{payload['mode']}] " + str(data.get("error") or data.get("detail") or f"HTTP {r.status_code}"))
    return data


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
        return "**Perhatian:** mode teroptimasi belum tersedia — bangun modul C++ dulu: `bash cpp/build.sh`."
    reps = 5
    med: dict[str, dict[str, float]] = {}
    top: dict[str, list[int]] = {}
    try:
        for m in modes:
            payload = {"question": q, "collection": collection, "mode": m, "generate": False, "stream": False,
                       "client": "ui-compare"}
            _compare_call(payload)  # pemanasan (muat snapshot/backend)
            runs = [_compare_call(payload) for _ in range(reps)]
            stages = sorted({k for r in runs for k in (r.get("timings_ms") or {}) if not k.startswith("queue")})
            med[m] = {k: sorted(r["timings_ms"].get(k, 0.0) for r in runs)[reps // 2] for k in stages}
            med[m]["__total"] = sorted(sum(v for k, v in r["timings_ms"].items() if not k.startswith("queue"))
                                       for r in runs)[reps // 2]
            top[m] = [s_["chunk_id"] for s_ in runs[-1].get("sources", [])]
    except Exception as exc:  # noqa: BLE001
        msg = _err(exc)
        if "out of memory" in msg.lower():
            return (f"**Gagal:** VRAM GPU penuh, jadi mode optimized tidak bisa memuat indeks koleksi `{collection}` "
                    "ke GPU. Server LLM sedang memakai sebagian besar VRAM.\n\nPakai koleksi `ugm`, atau jalankan demo "
                    "tanpa server LLM (perbandingan ini hanya mengukur retrieval) untuk membandingkan koleksi sintetis "
                    "besar.\n\n<sub>Detail: " + msg + "</sub>")
        return f"**Gagal:** {msg}"
    b, o = med["baseline"], med["optimized"]
    rows = ["| Tahap | Baseline | Optimized | Speedup |", "|---|---:|---:|---:|"]
    for k in [k for k in b if k != "__total"]:
        sp = b[k] / o[k] if o.get(k) else None
        rows.append(f"| {STAGE_LABELS.get(k, k)} | {b[k]:,.1f} ms | {o.get(k, 0):,.1f} ms | "
                    f"{'-' if sp is None else f'{sp:,.1f}×'} |")
    sp_total = b["__total"] / o["__total"] if o["__total"] else 0
    rows.append(f"| **Total retrieval** | **{b['__total']:,.1f} ms** | **{o['__total']:,.1f} ms** | **{sp_total:,.1f}×** |")
    same = "**identik**" if top["baseline"] == top["optimized"] else "**berbeda** (cek urutan skor seri)"
    head = (f'<div class="cmp-head">Optimized {sp_total:,.1f}× lebih cepat</div>'
            f'<div class="cmp-sub">Total retrieval {b["__total"]:,.1f} ms → {o["__total"]:,.1f} ms</div>\n\n')
    return (head + f"**Pertanyaan:** {q}\n\nMedian {reps} kali, koleksi `{collection}`, tanpa antrean & tanpa LLM.\n\n"
            + "\n".join(rows)
            + f"\n\nSumber konteks yang dipilih: {same}.\n\n<sub>Generasi LLM memakai model & server yang sama di kedua mode; "
            "keunggulannya (batching beberapa permintaan sekaligus) terlihat saat uji beban.</sub>")


# ====================================================================== knowledge base
def kb_view(collection: str):
    try:
        d = api_get(f"/api/kb/{collection}/documents")
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}", []
    st = d["stats"]
    idx = st.get("index") or {}
    t = idx.get("timings_ms") or {}
    info = [
        f"### Koleksi `{st['name']}`" + (" — **SINTETIS**" if st["synthetic"] else ""),
        f"{st['description']}",
        f"- Dokumen: **{st['n_documents']}** · Chunk: **{st['n_chunks']}**",
        f"- Indeks: {idx.get('n_chunks', 0)} chunk · model `{idx.get('embed_model', '-')}` · dibangun {idx.get('built_at', '-')}"
        + (" · **perlu dibangun ulang**" if st["index_stale"] else ""),
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
        return "**Perhatian:** pilih minimal satu file.", *kb_view(collection)
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
        return f"**Gagal:** {_err(exc)}", *kb_view(collection)
    finally:
        for _, (_, fh) in handles:
            fh.close()
    lines = []
    for x in res["results"]:
        name = x.get("title") or x.get("filename")
        if x["status"] == "ok":
            lines.append(f"- **{name}** — {x['n_chunks']} chunk, {x['n_chars']:,} karakter")
        else:
            lines.append(f"- **{name}** — gagal: {x.get('reason') or x.get('error')}")
        for w in x.get("warnings") or []:
            lines.append(f"  - Perhatian: {w}")
    return "\n".join(lines) + _fmt_index(res.get("index")), *kb_view(collection)


def url_fn(collection: str, url: str, title: str, category: str):
    if not (url or "").startswith(("http://", "https://")):
        return "**Perhatian:** masukkan URL lengkap (http/https).", *kb_view(collection)
    try:
        r = client.post(f"/api/kb/{collection}/url", json={"url": url, "title": title or None, "category": category or None})
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}", *kb_view(collection)
    x = res["result"]
    msg = (f"Berhasil: **{x.get('title')}** — {x.get('n_chunks')} chunk" if x["status"] == "ok"
           else f"Gagal: {x.get('reason') or x.get('error')}")
    for w in x.get("warnings") or []:
        msg += f"\n- Perhatian: {w}"
    return msg + _fmt_index(res.get("index")), *kb_view(collection)


def chunks_fn(collection: str, doc_id):
    if not doc_id:
        return "Masukkan ID dokumen."
    try:
        d = api_get(f"/api/kb/{collection}/documents/{int(doc_id)}/chunks", limit=30)
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}"
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
        return "**Perhatian:** centang konfirmasi dulu untuk menghapus.", *kb_view(collection), False
    try:
        r = client.delete(f"/api/kb/{collection}/documents/{int(doc_id)}")
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}", *kb_view(collection), False
    return f"Dokumen {int(doc_id)} dihapus ({res['deleted_chunks']} chunk)." + _fmt_index(res.get("index")), *kb_view(collection), False


def rebuild_fn(collection: str, full: bool):
    try:
        r = client.post(f"/api/kb/{collection}/rebuild", json={"full": full, "mode": "baseline"})
        r.raise_for_status()
        meta = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}", *kb_view(collection)
    return ("Rebuild " + ("penuh" if full else "inkremental") + " selesai." + _fmt_index(meta)), *kb_view(collection)


def create_collection_fn(name: str, desc: str, synthetic: bool):
    try:
        r = client.post("/api/collections", json={"name": (name or "").strip(), "description": desc or "", "synthetic": synthetic})
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}", gr.update()
    return f"Koleksi `{name}` dibuat.", gr.update(choices=collection_choices(), value=name.strip())


# ====================================================================== monitor & uji beban
LOADTEST: dict[str, Any] = {"proc": None, "started": None, "cfg": None, "prefix": None}


def metrics_md() -> str:
    try:
        m = api_get("/api/metrics", window=60)
    except Exception as exc:  # noqa: BLE001
        return f"**Gagal:** {_err(exc)}"
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
        return f"**Berjalan** — {cfg['users']} pengguna virtual, {elapsed:.0f}/{cfg['duration']} detik"
    from loadtest.run import parse_locust_csv

    res = parse_locust_csv(LOADTEST["prefix"])
    e2e, ttft = res.get("e2e") or {}, res.get("ttft") or {}
    done = e2e.get("count", 0) - e2e.get("failures", 0)
    return (f"**Selesai** ({cfg['users']} pengguna, {cfg['duration']} detik) — selesai {done}, gagal {e2e.get('failures', 0)}, "
            f"throughput {done / max(1, cfg['duration']):.2f} QPS, E2E p50/p95 {e2e.get('p50')}/{e2e.get('p95')} ms, "
            f"TTFT p50/p95 {ttft.get('p50')}/{ttft.get('p95')} ms. CSV: `{LOADTEST['prefix']}_stats.csv`")


def start_loadtest(users, duration, generate: bool, mode: str, collection: str, questions: str):
    p = LOADTEST["proc"]
    if p is not None and p.poll() is None:
        return "**Perhatian:** uji beban masih berjalan."
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
        return "Uji beban dihentikan."
    return loadtest_status()


# ====================================================================== layout
# Tampilan mengikuti template presentasi UGM (versi terang, "KKI Week 12"): latar putih, navy #074364,
# pita abu-abu #F4F5FB, foto pegunungan bersalju, kartu logo dengan garis navy, watermark emblem, catatan
# tempel kuning, dan kaki slide "www.ugm.ac.id | Locally Rooted, Globally Respected | logo".
ASSETS = Path(__file__).with_name("assets")


def _data_uri(path: Path, mime: str) -> str:
    try:
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ""


LOGO_HORIZONTAL = _data_uri(ASSETS / "ugm-logo-horizontal-navy.png", "image/png")
LOGO_STACKED = _data_uri(ASSETS / "ugm-logo-stacked-navy.png", "image/png")
MOUNTAIN = _data_uri(ASSETS / "ugm-mountain.jpg", "image/jpeg")
EMBLEM = _data_uri(ASSETS / "ugm-emblem-navy.png", "image/png")


def _font_faces() -> str:
    """Judul template memakai Proxima Nova (font komersial, tidak disertakan). Bila punya lisensinya,
    taruh berkasnya (nama mengandung "proxima") di ui/assets/fonts/ agar ditanam ke CSS; bila font itu
    terpasang di sistem, browser memakainya langsung. Tanpa keduanya dipakai Lato."""
    fmt = {".otf": "opentype", ".ttf": "truetype", ".woff": "woff", ".woff2": "woff2"}
    mime = {".otf": "font/otf", ".ttf": "font/ttf", ".woff": "font/woff", ".woff2": "font/woff2"}
    folder = ASSETS / "fonts"
    files = sorted(f for f in folder.glob("*") if f.suffix.lower() in fmt) if folder.exists() else []
    css = []
    for f in files:
        n = f.stem.lower().replace("-", "").replace("_", "").replace(" ", "")
        if "proxima" not in n:
            continue
        weight = 800 if ("extrabold" in n or "black" in n) else 700 if "bold" in n else 600 if "semibold" in n \
            else 300 if "light" in n else 400
        style = "italic" if ("italic" in n or n.endswith("it")) else "normal"
        css.append(f"@font-face {{ font-family: 'Proxima Nova'; font-weight: {weight}; font-style: {style}; "
                   f"font-display: swap; src: url({_data_uri(f, mime[f.suffix.lower()])}) "
                   f"format('{fmt[f.suffix.lower()]}'); }}")
    return "\n".join(css)


BRAND_HTML = (
    f'<div class="brand"><img class="brand-logo" src="{LOGO_HORIZONTAL}" alt="Universitas Gadjah Mada">'
    '<div class="brand-sub">Asisten Informasi Akademik</div></div>'
)
SLIDE_HEAD_HTML = (
    '<div class="slide-head"><span class="sh-stub"></span>'
    '<span class="sh-title">Asisten Informasi Akademik UGM</span>'
    '<span class="sh-sub">Pipeline RAG dengan akselerasi OpenMP &amp; CUDA</span>'
    '<span class="sh-burger" aria-hidden="true"></span></div>'
)
SLIDE_FOOT_HTML = (
    '<div class="slide-foot"><div class="sf-url">www.ugm.ac.id</div>'
    '<div class="sf-tag">Locally Rooted, Globally Respected</div>'
    f'<img class="sf-logo" src="{LOGO_HORIZONTAL}" alt="Universitas Gadjah Mada"></div>'
)
NOTE_HTML = ('<div class="sticky-note"><b>Catatan</b>Prototipe proyek mata kuliah HPC, bukan layanan resmi UGM. '
             'Selalu cek ulang jawaban ke dokumen sumbernya.</div>')
COMPARE_HINT = ("_Tulis pertanyaan lalu tekan **Bandingkan mode** untuk mengukur pertanyaan yang sama "
                "di mode baseline dan optimized._")

# Palet terang dari template: navy #074364, pita #F4F5FB, abu-abu #E9EDEE, teks #1A1A1A / #595959,
# biru muda #6AA4C8 (fokus), oranye #EB5600 (tombol berbahaya).
PALETTE = {
    "body_background_fill": "#ffffff", "body_text_color": "#1a1a1a", "body_text_color_subdued": "#5f6b73",
    "background_fill_primary": "#ffffff", "background_fill_secondary": "#f4f5fb",
    "block_background_fill": "#ffffff", "block_border_color": "#e1e6ee", "block_border_width": "1px",
    "block_label_background_fill": "transparent", "block_label_text_color": "#5f6b73",
    "block_title_text_color": "#074364", "block_info_text_color": "#5f6b73", "block_radius": "4px",
    "block_shadow": "none", "panel_background_fill": "#ffffff", "panel_border_color": "#e1e6ee",
    "border_color_primary": "#e1e6ee", "border_color_accent": "#074364", "color_accent": "#074364",
    "color_accent_soft": "#e7eef4", "input_background_fill": "#ffffff", "input_background_fill_focus": "#ffffff",
    "input_border_color": "#d3dbe3", "input_border_color_focus": "#074364", "input_placeholder_color": "#8a969e",
    "input_radius": "4px", "input_shadow": "none", "input_shadow_focus": "0 0 0 3px rgba(106, 164, 200, .25)",
    "button_primary_background_fill": "#074364", "button_primary_background_fill_hover": "#0a5784",
    "button_primary_text_color": "#ffffff", "button_primary_border_color": "#074364",
    "button_secondary_background_fill": "#f4f5fb", "button_secondary_background_fill_hover": "#e9edf5",
    "button_secondary_text_color": "#074364", "button_secondary_border_color": "#d3dbe3",
    "button_cancel_background_fill": "#EB5600", "button_cancel_background_fill_hover": "#c94a00",
    "button_cancel_text_color": "#ffffff", "button_cancel_border_color": "#EB5600",
    "button_large_radius": "4px", "button_medium_radius": "4px", "button_small_radius": "4px",
    "table_even_background_fill": "#ffffff", "table_odd_background_fill": "#f7f8fc",
    "table_border_color": "#e1e6ee", "table_text_color": "#1a1a1a", "code_background_fill": "#f4f5fb",
    "link_text_color": "#074364", "link_text_color_hover": "#0a5784", "link_text_color_visited": "#074364",
    "accordion_text_color": "#074364", "loader_color": "#074364", "chatbot_text_size": "16px",
    "checkbox_background_color": "#ffffff", "checkbox_border_color": "#9fb0bd",
    "checkbox_background_color_selected": "#074364", "checkbox_border_color_selected": "#074364",
    "checkbox_label_background_fill": "#ffffff", "checkbox_label_background_fill_selected": "#f4f5fb",
    "checkbox_label_border_color": "#d3dbe3", "checkbox_label_text_color": "#1a1a1a",
    "checkbox_label_text_color_selected": "#074364", "stat_background_fill": "#074364",
}
FONT_STACK = ["Proxima Nova", "Lato", "Open Sans", "Noto Sans", "Segoe UI", "system-ui", "sans-serif"]
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=FONT_STACK,
    font_mono=["JetBrains Mono", "Fira Mono", "ui-monospace", "monospace"],
)
# Template hanya punya versi terang: varian _dark diisi palet yang sama agar tampilan tidak berubah
# walau browser/OS memakai mode gelap.
THEME.set(**PALETTE, **{f"{k}_dark": v for k, v in PALETTE.items() if hasattr(THEME, f"{k}_dark")})

HEAD = ""
# Gradio mengukur tinggi textarea sekali saat halaman dimuat (sebelum tata letak & font stabil) sehingga
# kotak tanya bisa tampil setinggi max_lines; picu ulang perhitungannya setelah halaman siap.
JS = """() => {
  document.body.classList.remove('dark');
  document.documentElement.classList.remove('dark');
  const fit = () => {
    const ta = document.querySelector('#composer-input textarea');
    if (ta) ta.dispatchEvent(new Event('input', { bubbles: true }));
  };
  [300, 1200, 3000].forEach((t) => setTimeout(fit, t));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
  window.addEventListener('resize', fit);
}"""

CSS = _font_faces() + """
:root { --font: 'Proxima Nova', 'Lato', 'Open Sans', 'Noto Sans', system-ui, sans-serif;
        --navy: #074364; --navy2: #0a5784; --mist: #F4F5FB; --stub: #E9EDEE; --ink: #1a1a1a; --ink2: #33464f;
        --muted: #5f6b73; --line: #e1e6ee; --line2: #d3dbe3; --sky: #6AA4C8; --note: #F5EA8D; --side-w: 280px;
        --foot-h: 38px; }
body, .gradio-container, .gradio-container * { font-family: var(--font); }
code, pre, .gradio-container code { font-family: 'JetBrains Mono', 'Fira Mono', monospace !important; }
body, .gradio-container { background: #ffffff !important; color: var(--ink); }
footer { display: none !important; }
.gradio-container { max-width: none !important; padding: 0 0 calc(var(--foot-h) + 14px) !important; }
/* watermark emblem seperti slide isi template */
.gradio-container::after { content: ""; position: fixed; right: -80px; top: 90px; width: 36vw; max-width: 500px;
    aspect-ratio: 617 / 640; background: url("__EMBLEM__") no-repeat center / contain; opacity: .045;
    pointer-events: none; z-index: 0; }
.gradio-container > .main, #main-tabs { position: relative; z-index: 1; }

/* ---------- kepala slide: pita abu-abu + blok navy + judul */
#slide-head { padding: 0 !important; }
.slide-head { display: flex; align-items: center; gap: 14px; padding: 22px 32px 10px 0; }
.sh-stub { flex: none; width: 64px; height: 26px;
    background: linear-gradient(to right, var(--stub) 0 calc(100% - 13px), var(--navy) calc(100% - 13px) 100%); }
.sh-title { color: var(--navy); font-weight: 700; font-size: 19px; letter-spacing: .1px; white-space: nowrap; }
.sh-sub { color: var(--muted); font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sh-burger { margin-left: auto; flex: none; width: 24px; height: 7px; border-top: 2px solid #b9c3ca;
    border-bottom: 2px solid #b9c3ca; }

/* ---------- kaki slide */
#slide-foot { padding: 0 !important; }
.slide-foot { position: fixed; left: 0; right: 0; bottom: 0; height: var(--foot-h); display: flex; align-items: center;
    background: var(--mist); border-top: 1px solid #e6e9f2; z-index: 1000; }
.slide-foot .sf-url { flex: none; width: var(--side-w); height: 100%; display: flex; align-items: center;
    justify-content: center; background: var(--navy); color: #ffffff; font-weight: 700; font-size: 14.5px; letter-spacing: .2px; }
.slide-foot .sf-tag { flex: 1; text-align: center; color: var(--navy); font-size: 14.5px; }
.slide-foot .sf-logo { flex: none; height: 27px; width: auto; margin-right: 24px; }

/* ---------- sidebar */
.sidebar, #sidebar { background: var(--mist) !important; border-right: 1px solid #e3e7ef !important; }
#sidebar .sidebar-content { padding-bottom: calc(var(--foot-h) + 12px) !important; scrollbar-width: thin;
    scrollbar-color: #cfd6df transparent; }
#sidebar .brand { padding: 6px 6px 12px 4px; }
#sidebar .brand-logo { width: 212px; height: auto; display: block; }
#sidebar .brand-sub { margin-top: 12px; padding-left: 2px; color: var(--navy); font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 800; }
#sidebar button.nav { justify-content: flex-start !important; text-align: left !important; background: transparent !important;
    border: none !important; box-shadow: none !important; color: var(--ink2) !important; font-weight: 400;
    padding: 8px 12px !important; min-height: 36px; border-radius: 2px !important; font-size: 14.5px; }
#sidebar button.nav:hover { background: #e9edf5 !important; color: var(--navy) !important; }
#sidebar button.nav.primary { background: #ffffff !important; color: var(--navy) !important; font-weight: 700;
    box-shadow: inset 4px 0 0 var(--navy), 0 1px 3px rgba(7, 67, 100, .08) !important; }
#sidebar button.nav-new { color: #ffffff !important; background: var(--navy) !important; font-weight: 700;
    justify-content: center !important; margin-bottom: 10px; box-shadow: none !important; }
#sidebar button.nav-new:hover { background: var(--navy2) !important; color: #ffffff !important; }
#sidebar .sb-section { display: flex; align-items: center; gap: 8px; margin: 16px 12px 4px; font-size: 11px;
    color: var(--navy); font-weight: 800; letter-spacing: 1.6px; text-transform: uppercase; }
#sidebar .sb-section::before { content: ""; width: 5px; height: 12px; background: var(--navy); }
#sidebar .sb-hint { margin: 2px 12px 0; font-size: 12.5px; color: var(--muted); font-style: italic; }
#recents { background: transparent !important; border: none !important; padding: 0 !important; }
#recents .wrap { flex-direction: column !important; gap: 0 !important; }
#recents label { background: transparent !important; border: none !important; box-shadow: none !important;
    padding: 6px 12px !important; border-radius: 2px !important; color: var(--ink2) !important; width: 100%;
    font-size: 13.5px; cursor: pointer; }
#recents label:hover { background: #e9edf5 !important; color: var(--navy) !important; }
#recents label span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
#recents input[type=radio] { display: none !important; }
#sb-foot { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px; }
#status .st-row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); padding: 2px 6px; }
#status .st-row b { color: var(--navy); font-weight: 700; min-width: 30px; }
#status .st-detail { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#status .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
#status .dot.ok { background: #1A9988; } #status .dot.bad { background: #EB5600; } #status .dot.warn { background: #d9a400; }
/* catatan tempel kuning seperti di slide isi template */
.sticky-note { margin: 12px 12px 6px 8px; padding: 12px 14px 16px; background: linear-gradient(180deg, #f7ee9c, var(--note));
    color: #3d3a22; font-size: 12px; line-height: 1.5; transform: rotate(-1.2deg);
    box-shadow: 1px 6px 10px -4px rgba(0, 0, 0, .28); border-radius: 1px 1px 14px 1px / 1px 1px 22px 1px; }
.sticky-note b { display: block; font-size: 11px; letter-spacing: 1.4px; text-transform: uppercase; margin-bottom: 4px; }

/* ---------- tab utama (navigasi lewat sidebar) */
#main-tabs > .tab-wrapper, #main-tabs > div[role="tablist"], #main-tabs .tab-container { display: none !important; }
#main-tabs { border: none !important; background: transparent !important; }
#main-tabs > .tabitem { border: none !important; padding: 0 !important; background: transparent !important; }

/* ---------- chat */
#chat-row { padding: 0 32px 0 50px; gap: 26px; }
#chat-col { max-width: 880px; margin: 0 auto; }
/* latar halaman chat = pegunungan bersalju slide judul, menempel di atas kaki slide; langit memudar ke putih.
   Saat percakapan berjalan (#chat ada di DOM) gambarnya diredupkan agar teks jawaban tetap mudah dibaca. */
#chat-tab::before { content: ""; position: fixed; left: 0; right: 0; bottom: var(--foot-h); height: 64vh; z-index: -1;
    pointer-events: none; transition: opacity .4s ease;
    background: linear-gradient(to bottom, #ffffff 0%, rgba(255, 255, 255, 0) 46%), url("__MOUNTAIN__") center bottom / cover no-repeat; }
#chat-tab:has(#chat)::before { opacity: .38; }
/* sampul = slide judul: kartu logo + garis navy, judul rata kanan */
#hero { margin-top: 2vh; }
#hero .hero { position: relative; display: flex; justify-content: space-between; gap: 28px; margin-bottom: 26px; }
#hero .hero-card { flex: none; align-self: flex-start; width: 170px; height: 214px; margin-left: 22px; display: flex;
    align-items: center; justify-content: center; background: var(--mist); border-bottom: 9px solid var(--navy);
    box-shadow: 0 2px 12px rgba(7, 67, 100, .10); }
#hero .hero-card img { width: 118px; height: auto; }
#hero .hero-copy { text-align: right; padding: 14px 22px 40px 0; max-width: 640px; }
#hero .hero-eyebrow { color: var(--navy); font-weight: 800; font-size: 13px; letter-spacing: 1px; text-transform: uppercase; }
#hero .hero-text { color: var(--navy); font-weight: 800; font-size: 34px; line-height: 1.18; margin-top: 16px;
    letter-spacing: -.2px; }
#hero .hero-sub { color: #2f4a5c; font-size: 15.5px; margin-top: 14px; line-height: 1.5; }
#composer { background: #ffffff !important; border: 1px solid var(--line2) !important; border-radius: 6px !important;
    box-shadow: 0 8px 26px rgba(7, 67, 100, .10) !important; padding: 6px 8px 8px !important; overflow: visible !important; }
#composer:focus-within { border-color: var(--navy) !important; }
#composer > div, #composer .form { background: transparent !important; border: none !important; box-shadow: none !important; }
#composer-input, #composer-input textarea { background: transparent !important; border: none !important; box-shadow: none !important; }
#composer-input textarea { font-size: 16px !important; color: var(--ink) !important; min-height: 56px; padding: 12px 10px 4px !important; }
#composer-bar { align-items: center; gap: 6px; background: transparent !important; flex-wrap: nowrap; }
#composer-bar .spacer { flex: 1; }
#pick-col, #pick-mode { background: transparent !important; border: none !important; box-shadow: none !important; }
#pick-col .wrap, #pick-mode .wrap, #pick-col input, #pick-mode input { background: transparent !important; border: none !important;
    box-shadow: none !important; color: var(--ink2) !important; font-size: 13.5px !important; }
#pick-col .wrap:hover, #pick-mode .wrap:hover { background: var(--mist) !important; border-radius: 4px; }
#pick-mode input, #pick-col input { text-align: left; padding-right: 22px !important; }
#send-btn { min-width: 36px !important; max-width: 36px !important; height: 36px; padding: 0 !important; border-radius: 4px !important;
    font-size: 18px; font-weight: 700; }
#under-composer { align-items: center; margin-top: 8px; gap: 10px; }
#mode-desc, #mode-desc * { color: var(--muted) !important; font-size: 12.5px !important; font-style: italic; }
#mode-desc { padding: 4px 10px !important; background: rgba(255, 255, 255, .85) !important; }
#mode-desc p { margin: 0; line-height: 1.45; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; }
button.chip { background: #ffffff !important; border: 1px solid var(--line2) !important; color: var(--navy) !important;
    border-radius: 4px !important; font-weight: 700; }
button.chip:hover { border-color: var(--navy) !important; background: var(--mist) !important; }
#examples { margin-top: 28px; }
#examples .ex-title { display: flex; align-items: center; gap: 8px; color: var(--navy); font-size: 11px; font-weight: 800;
    letter-spacing: 1.6px; text-transform: uppercase; margin: 0 4px 10px; }
#examples .ex-title::before { content: ""; width: 5px; height: 12px; background: var(--navy); }
#ex-grid { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; counter-reset: ex; }
#ex-grid > * { min-width: 0 !important; width: auto !important; }
button.ex-card { background: #ffffff !important; border: 1px solid var(--line) !important; color: var(--ink2) !important;
    text-align: left !important; justify-content: flex-start !important; align-items: flex-start !important;
    flex-direction: column !important; white-space: normal !important; min-height: 104px; padding: 14px !important;
    border-radius: 2px !important; font-weight: 400; font-size: 13.5px; line-height: 1.45; box-shadow: none !important; }
button.ex-card::before { counter-increment: ex; content: "0" counter(ex); display: block; color: var(--navy); font-weight: 800;
    font-size: 13px; margin-bottom: 6px; }
button.ex-card:hover { border-color: var(--navy) !important; color: var(--ink) !important; background: var(--mist) !important; }
.disclaimer { display: table; margin: 12px auto 0; padding: 2px 12px; background: rgba(255, 255, 255, .82); color: var(--muted);
    font-size: 12px; font-style: italic; }

/* gelembung chat: pengguna dalam pita abu-abu dengan blok navy, asisten teks polos */
#chat, #chat > div, #chat .bubble-wrap, #chat .wrapper { background: transparent !important; border: none !important;
    box-shadow: none !important; }
#chat .message-row { margin: 6px 0 !important; }
#chat .user-row .message, #chat .message.user { background: var(--mist) !important; border: none !important;
    box-shadow: inset -4px 0 0 var(--navy) !important; color: var(--ink) !important; border-radius: 2px !important;
    padding: 10px 16px !important; }
#chat .bot-row .message, #chat .message.bot { background: transparent !important; border: none !important; box-shadow: none !important;
    padding: 4px 2px !important; }
#chat .message.bot .prose, #chat .bot-row .prose { font-size: 16px !important; line-height: 1.7 !important; color: var(--ink) !important;
    text-align: justify; }
#chat .prose strong { color: #0d1b24; }
#chat .prose a { color: var(--navy); }
#chat .placeholder, #chat .avatar-container { display: none !important; }

/* ---------- panel kanan */
#side-panel { position: sticky; top: 12px; align-self: flex-start; max-height: calc(100vh - var(--foot-h) - 24px);
    overflow-y: auto; overflow-x: hidden; flex-wrap: nowrap !important; }
#side-panel > .side-acc { flex: none !important; }
#cmp-acc .cmp-head { color: var(--navy); font-size: 22px; font-weight: 800; line-height: 1.2; margin: 2px 0 4px; }
#cmp-acc .cmp-sub { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
#cmp-acc table { font-size: 12.5px; width: 100%; }
#cmp-acc th, #cmp-acc td { padding: 4px 6px !important; word-break: normal !important; overflow-wrap: normal !important; }
#cmp-acc th:not(:first-child), #cmp-acc td:not(:first-child) { white-space: nowrap; }
#side-panel .side-acc { background: #ffffff !important; border: 1px solid var(--line) !important; border-radius: 2px !important;
    box-shadow: inset 4px 0 0 var(--navy) !important; margin-bottom: 12px; padding-left: 6px !important; }
#side-panel .side-acc > button, #side-panel .side-acc .label-wrap { font-weight: 700; color: var(--navy) !important; }
#side-panel .sources { max-height: 420px; overflow-y: auto; }
#side-panel table { font-size: 13px; }

/* ---------- halaman KB & monitor: judul bagian bernomor seperti slide isi */
.sect-title { color: var(--navy); font-size: 26px; font-weight: 800; margin: 14px 0 4px; letter-spacing: -.2px; }
.sect-title .sect-num { margin-right: 6px; }
.page-sub { color: var(--muted); font-size: 14.5px; margin-bottom: 16px; }
#kb-page, #mon-page { padding: 0 40px 24px 50px; max-width: 1240px; }
#kb-page .block, #mon-page .block { box-shadow: none !important; }

/* ---------- halaman hasil speedup: kartu angka besar + grafik + tabel */
#res-page { padding: 0 40px 24px 50px; max-width: 1280px; }
#res-page .block { box-shadow: none !important; }
.stat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 4px 0 10px; }
.stat { background: #ffffff; border: 1px solid var(--line); box-shadow: inset 0 -6px 0 var(--navy);
    padding: 16px 18px 20px; }
.stat-v { color: var(--navy); font-size: 42px; font-weight: 800; line-height: 1; letter-spacing: -.5px; }
.stat-l { color: var(--ink); font-weight: 700; font-size: 15px; margin-top: 10px; }
.stat-d { color: var(--muted); font-size: 12.5px; margin-top: 4px; line-height: 1.4; }
.res-meta { color: var(--muted); font-size: 12.5px; font-style: italic; }
.res-h { display: flex; align-items: center; gap: 10px; color: var(--navy); font-weight: 800; font-size: 19px;
    margin: 30px 0 2px; }
.res-h::before { content: ""; flex: none; width: 6px; height: 18px; background: var(--navy); }
.res-sub { color: var(--muted); font-size: 13.5px; margin: 2px 0 10px 16px; }
.res-md table { font-size: 13px; width: auto !important; }
.res-md th, .res-md td { padding: 5px 12px !important; word-break: normal !important; overflow-wrap: normal !important; }
.res-md td { white-space: nowrap; }
.res-md code { font-size: 12px; }
.res-fig { border: 1px solid var(--line) !important; background: #ffffff !important; max-width: 1080px; margin: 6px 0; }
.res-fig img { object-fit: contain !important; }
""".replace("__EMBLEM__", EMBLEM).replace("__MOUNTAIN__", MOUNTAIN)


def _page_header(num: str, title: str, sub: str) -> str:
    return f'<div class="sect-title"><span class="sect-num">{num}</span>{title}</div><div class="page-sub">{sub}</div>'


def _res_head(title: str, sub: str = "") -> None:
    gr.HTML(f'<div class="res-h">{title}</div>' + (f'<div class="res-sub">{sub}</div>' if sub else ""))


def build_results_page() -> None:
    """Halaman 04: hasil pengukuran sebelum vs sesudah optimasi, dibaca dari berkas di results/."""
    r = bench.load()
    figs = r["figures"]

    def fig(*names: str) -> None:
        # satu grafik per baris: tiap gambar sudah berisi dua panel, berdampingan jadi terlalu kecil dibaca
        for path in (figs[n] for n in names if n in figs):
            _kw(gr.Image, value=path, show_label=False, interactive=False, buttons=["fullscreen"],
                elem_classes=["res-fig"])

    with gr.Column(elem_id="res-page"):
        gr.HTML(_page_header("04", "Hasil Speedup",
                             "Pengukuran sebelum (baseline) dan sesudah optimasi pada mesin yang sama. Angka dibaca "
                             "langsung dari berkas di <code>results/</code>; ulangi pengukuran dengan "
                             "<code>bash loadtest/run_compare.sh</code> lalu muat ulang UI."))
        if not r["title"]:
            gr.Markdown("_Belum ada `results/comparison.md`. Jalankan `bash loadtest/run_compare.sh` "
                        "lalu `python -m tools.compare` untuk membuatnya._")
            return
        gr.HTML(r["cards"])
        gr.HTML(f'<div class="res-meta">{r["title"]} · RTX 5050 Laptop 8 GB · Core i5-13450HX · '
                'kualitas retrieval identik (lihat bagian terakhir)</div>')

        _res_head("Retrieval di bawah beban: korpus sintetis 10⁶ chunk",
                  "Uji beban Locust tanpa LLM, rata-rata 3 pengulangan per titik. Di skala ini kernel CUDA dan "
                  "OpenMP paling terasa.")
        fig("fig2_retrieval_qps", "fig1_retrieval_p95")
        gr.Markdown(r["big"], elem_classes=["res-md"])

        _res_head("Retrieval di bawah beban: KB resmi UGM (1.920 chunk)",
                  "Korpus kecil: pencarian sudah murah, percepatan datang dari reranking FP16 dan batching.")
        gr.Markdown(r["ugm"], elem_classes=["res-md"])

        _res_head("Analisis bottleneck: di mana waktunya habis?",
                  "Komposisi latensi per tahap dan ablation kumulatif (setiap langkah menambah satu optimasi).")
        fig("fig3_stage_breakdown", "fig4_ablation")
        if r["ablation"]:
            with gr.Accordion(f"Tabel ablation & hukum Amdahl ({r['ablation_name']})", open=False):
                gr.Markdown(r["ablation"], elem_classes=["res-md"])

        _res_head("Kernel CUDA & OpenMP",
                  "Micro-benchmark kernel similarity + radix-select (GPU) dan strong scaling BM25 (CPU).")
        fig("fig6_cuda_kernel", "fig5_bm25_scaling")
        if r["kernels"]:
            with gr.Accordion("Tabel micro-benchmark kernel", open=False):
                gr.Markdown(r["kernels"], elem_classes=["res-md"])

        _res_head("End-to-end dengan LLM",
                  "Pipeline lengkap sampai jawaban LLM. Throughput token dipakai karena panjang jawaban tiap "
                  "runtime berbeda.")
        fig("fig7_e2e_tokens")
        gr.Markdown(r["e2e"], elem_classes=["res-md"])
        if r["llm_probe"]:
            with gr.Accordion("Server LLM: satu vs empat permintaan sekaligus", open=False):
                gr.Markdown(r["llm_probe"], elem_classes=["res-md"])

        _res_head("Build indeks dari nol", "Parsing paralel (pool proses) dan embedding batch FP16.")
        fig("fig8_build")
        gr.Markdown(r["build"], elem_classes=["res-md"])

        if r["quality"]:
            _res_head("Kualitas tetap sama", "Recall@k dan MRR mode baseline vs optimized (tools.eval_recall).")
            gr.Markdown(r["quality"], elem_classes=["res-md"])


def build_ui() -> gr.Blocks:
    modes, _ = mode_choices()
    cols = collection_choices()
    default_col = (CFG.get("kb") or {}).get("default_collection", "ugm")
    default_col = default_col if any(v == default_col for _, v in cols) else cols[0][1]
    with gr.Blocks(title="Asisten Informasi Akademik UGM") as demo:
        recents_state = gr.State([])

        # ------------------------------------------------------------ sidebar
        with gr.Sidebar(open=True, width=280, elem_id="sidebar"):
            gr.HTML(BRAND_HTML)
            new_btn = gr.Button("+  Percakapan baru", elem_classes=["nav", "nav-new"])
            nav_chat = gr.Button("01   Chat", variant="primary", elem_classes=["nav"])
            nav_kb = gr.Button("02   Knowledge Base", variant="secondary", elem_classes=["nav"])
            nav_mon = gr.Button("03   Monitor & Uji Beban", variant="secondary", elem_classes=["nav"])
            nav_res = gr.Button("04   Hasil Speedup", variant="secondary", elem_classes=["nav"])
            gr.HTML('<div class="sb-section">Terbaru</div>')
            recents_hint = gr.HTML('<div class="sb-hint">Pertanyaan di sesi ini akan muncul di sini.</div>')
            recents = gr.Radio(choices=[], value=None, show_label=False, container=False, elem_id="recents")
            with gr.Column(elem_id="sb-foot"):
                status = gr.HTML(status_html(), elem_id="status")
                refresh = gr.Button("Muat ulang status", size="sm", elem_classes=["nav"])
                gr.HTML(NOTE_HTML, elem_id="foot-note")

        _kw(gr.HTML, value=SLIDE_HEAD_HTML, elem_id="slide-head", padding=False)
        with gr.Tabs(selected="chat", elem_id="main-tabs") as tabs:
            # -------------------------------------------------------- chat
            with gr.Tab("Chat", id="chat", elem_id="chat-tab"):
                with gr.Row(elem_id="chat-row", equal_height=False):
                    with gr.Column(scale=3, elem_id="chat-col"):
                        hero = gr.HTML(greeting_html(), elem_id="hero")
                        chatbot = _kw(gr.Chatbot, visible=False, elem_id="chat", show_label=False, height="calc(100vh - 430px)",
                                      layout="bubble", buttons=["copy", "copy_all"])
                        with gr.Group(elem_id="composer"):
                            msg = gr.Textbox(placeholder="Tanya apa saja tentang informasi akademik UGM…", show_label=False,
                                             container=False, lines=1, max_lines=8, autofocus=True, elem_id="composer-input")
                            with gr.Row(elem_id="composer-bar"):
                                collection = gr.Dropdown(choices=cols, value=default_col, show_label=False, container=False,
                                                         elem_id="pick-col", min_width=230, scale=0)
                                gr.HTML('<div class="spacer"></div>')
                                mode = gr.Dropdown(choices=modes, value="baseline", show_label=False, container=False,
                                                   elem_id="pick-mode", min_width=160, scale=0)
                                send = gr.Button("↑", variant="primary", elem_id="send-btn", min_width=36, scale=0)
                        with gr.Row(elem_id="under-composer"):
                            mode_desc = gr.Markdown("", elem_id="mode-desc")
                            compare = gr.Button("Bandingkan mode", size="sm", elem_classes=["chip"], scale=0, min_width=160)
                        with gr.Column(elem_id="examples") as examples_box:
                            gr.HTML('<div class="ex-title">Contoh pertanyaan</div>')
                            with gr.Row(elem_id="ex-grid"):
                                ex_btns = [gr.Button(q, elem_classes=["ex-card"]) for q in EXAMPLES[:4]]
                        gr.HTML('<div class="disclaimer">Jawaban disusun hanya dari dokumen resmi di knowledge base — '
                                'periksa kembali sumbernya.</div>')
                    with gr.Column(scale=2, visible=False, elem_id="side-panel", min_width=320) as side:
                        with gr.Accordion("Sumber", open=True, elem_classes=["side-acc"]):
                            sources = gr.Markdown("_Sumber jawaban akan muncul di sini._", elem_classes=["sources"])
                        with gr.Accordion("Latensi per tahap", open=True, elem_classes=["side-acc"]):
                            timings = gr.Markdown("")
                        with gr.Accordion("Baseline vs optimized", open=True, elem_id="cmp-acc", elem_classes=["side-acc"]):
                            comparison = gr.Markdown(COMPARE_HINT)

            # -------------------------------------------------------- knowledge base
            with gr.Tab("Knowledge Base", id="kb"):
                with gr.Column(elem_id="kb-page"):
                    gr.HTML(_page_header("02", "Knowledge Base",
                                         "Dokumen resmi yang menjadi satu-satunya sumber jawaban. Koleksi aktif mengikuti "
                                         "pilihan di kotak tanya."))
                    kb_info = gr.Markdown("")
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion("Tambah dokumen dari file", open=True):
                                up_files = gr.File(file_count="multiple", file_types=[".pdf", ".html", ".htm", ".txt", ".md"],
                                                   label="File (PDF/HTML/TXT/MD)")
                                up_title = gr.Textbox(label="Judul (opsional, hanya jika 1 file)")
                                up_cat = gr.Dropdown(CATEGORIES, label="Kategori", allow_custom_value=True)
                                up_url = gr.Textbox(label="URL sumber resmi (opsional, untuk sitasi)")
                                up_btn = gr.Button("Unggah & indeks", variant="primary")
                            with gr.Accordion("Tambah dari URL resmi", open=False):
                                url_in = gr.Textbox(label="URL (halaman HTML atau PDF)")
                                url_title = gr.Textbox(label="Judul (opsional)")
                                url_cat = gr.Dropdown(CATEGORIES, label="Kategori", allow_custom_value=True)
                                url_btn = gr.Button("Ambil & indeks", variant="primary")
                            with gr.Accordion("Buat koleksi baru", open=False):
                                new_name = gr.Textbox(label="Nama (huruf kecil, angka, - _)")
                                new_desc = gr.Textbox(label="Deskripsi")
                                new_syn = gr.Checkbox(label="Koleksi sintetis (data dummy untuk pengujian)")
                                new_col_btn = gr.Button("Buat koleksi")
                        with gr.Column():
                            kb_msg = gr.Markdown("")
                            with gr.Row():
                                rb_inc = gr.Button("Rebuild inkremental")
                                rb_full = gr.Button("Rebuild penuh (ukur waktu build)")
                            with gr.Row():
                                doc_id = gr.Number(label="ID dokumen", precision=0)
                                view_btn = gr.Button("Lihat chunk")
                            with gr.Row():
                                del_confirm = gr.Checkbox(label="Saya yakin ingin menghapus dokumen ini")
                                del_btn = gr.Button("Hapus dokumen", variant="stop")
                    docs = _kw(gr.Dataframe, headers=["ID", "Judul", "Kategori", "Chunk", "Tipe", "URL sumber", "Ditambahkan"],
                               interactive=False, wrap=True, max_height=500, label="Dokumen di koleksi")
                    chunk_view = gr.Markdown("")

            # -------------------------------------------------------- monitor & uji beban
            with gr.Tab("Monitor & Uji Beban", id="mon"):
                with gr.Column(elem_id="mon-page"):
                    gr.HTML(_page_header("03", "Monitor & Uji Beban",
                                         "Metrik server per mode, diperbarui setiap 2 detik. Jalankan pengguna virtual sambil "
                                         "tetap bertanya di halaman Chat."))
                    metrics = gr.Markdown(metrics_md())
                    with gr.Row():
                        lt_users = gr.Number(value=10, label="Pengguna virtual", precision=0)
                        lt_dur = gr.Number(value=60, label="Durasi (detik)", precision=0)
                        lt_gen = gr.Checkbox(value=True, label="Dengan generasi LLM")
                        lt_q = gr.Textbox(value="loadtest/questions_ugm.txt", label="File pertanyaan")
                    with gr.Row():
                        lt_start = gr.Button("Mulai uji beban (mode & koleksi dari kotak tanya)", variant="primary")
                        lt_stop = gr.Button("Hentikan", variant="stop")
                    lt_status = gr.Markdown(loadtest_status())
                    timer = gr.Timer(2.0)
                    timer.tick(lambda: (metrics_md(), loadtest_status()), outputs=[metrics, lt_status])

            # -------------------------------------------------------- hasil speedup (benchmark terukur)
            with gr.Tab("Hasil Speedup", id="hasil"):
                build_results_page()

        _kw(gr.HTML, value=SLIDE_FOOT_HTML, elem_id="slide-foot", padding=False)

        # ------------------------------------------------------------ event
        navs = [nav_chat, nav_kb, nav_mon, nav_res]
        refresh.click(refresh_header, outputs=[status, mode, collection])
        mode.change(on_mode_change, inputs=mode, outputs=[mode, mode_desc])
        for trigger in (send.click, msg.submit):
            trigger(user_submit, inputs=[msg, chatbot, recents_state],
                    outputs=[msg, chatbot, send, hero, examples_box, side, recents_state, recents, recents_hint],
                    queue=False).then(
                bot_respond, inputs=[chatbot, mode, collection], outputs=[chatbot, sources, timings]
            ).then(unlock_input, outputs=[msg, send], queue=False)
        new_btn.click(new_chat, outputs=[chatbot, hero, examples_box, side, sources, timings, comparison, msg, tabs, *navs])
        nav_chat.click(lambda: go("chat"), outputs=[tabs, *navs])
        nav_kb.click(lambda: go("kb"), outputs=[tabs, *navs])
        nav_mon.click(lambda: go("mon"), outputs=[tabs, *navs])
        nav_res.click(lambda: go("hasil"), outputs=[tabs, *navs])
        recents.input(pick_recent, inputs=recents, outputs=[msg, tabs, *navs])
        for b, q in zip(ex_btns, EXAMPLES[:4]):
            b.click(lambda q=q: gr.update(value=q), outputs=msg, queue=False)
        compare.click(lambda: (gr.update(visible=True), "_Mengukur kedua mode…_"), outputs=[side, comparison],
                      queue=False).then(compare_fn, inputs=[msg, chatbot, collection], outputs=comparison).then(
            # gulir panel kanan (bukan halaman) ke hasil; ditunda sampai hasil selesai dirender
            None, js="() => { const go = () => { const sp = document.querySelector('#side-panel'); "
                     "const el = document.querySelector('#cmp-acc'); "
                     "if (sp && el) sp.scrollTo({ top: el.offsetTop - 8, behavior: 'smooth' }); }; "
                     "[250, 900].forEach((t) => setTimeout(go, t)); }")

        collection.change(kb_view, inputs=collection, outputs=[kb_info, docs])
        up_btn.click(upload_fn, inputs=[collection, up_files, up_title, up_cat, up_url], outputs=[kb_msg, kb_info, docs])
        url_btn.click(url_fn, inputs=[collection, url_in, url_title, url_cat], outputs=[kb_msg, kb_info, docs])
        view_btn.click(chunks_fn, inputs=[collection, doc_id], outputs=chunk_view)
        del_btn.click(delete_fn, inputs=[collection, doc_id, del_confirm], outputs=[kb_msg, kb_info, docs, del_confirm])
        rb_inc.click(lambda c: rebuild_fn(c, False), inputs=collection, outputs=[kb_msg, kb_info, docs])
        rb_full.click(lambda c: rebuild_fn(c, True), inputs=collection, outputs=[kb_msg, kb_info, docs])
        new_col_btn.click(create_collection_fn, inputs=[new_name, new_desc, new_syn], outputs=[kb_msg, collection])
        lt_start.click(start_loadtest, inputs=[lt_users, lt_dur, lt_gen, mode, collection, lt_q], outputs=lt_status)
        lt_stop.click(stop_loadtest, outputs=lt_status)
        demo.load(kb_view, inputs=collection, outputs=[kb_info, docs])
        demo.load(lambda: (greeting_html(), status_html()), outputs=[hero, status])
        demo.load(on_mode_change, inputs=mode, outputs=[mode, mode_desc])
    return demo
