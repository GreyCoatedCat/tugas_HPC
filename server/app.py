"""FastAPI server.

Endpoint utama:
    POST /api/chat                      tanya-jawab (stream NDJSON atau JSON biasa)
    GET  /api/health | /api/modes | /api/metrics
    GET  /api/collections               daftar koleksi KB
    POST /api/collections               buat koleksi baru
    GET  /api/kb/{koleksi}/documents    daftar dokumen
    POST /api/kb/{koleksi}/documents    unggah file (PDF/HTML/TXT/MD) -> ingest -> indeks
    POST /api/kb/{koleksi}/url          ambil URL resmi -> ingest -> indeks
    DELETE /api/kb/{koleksi}/documents/{id}
    GET  /api/kb/{koleksi}/documents/{id}/chunks
    POST /api/kb/{koleksi}/rebuild      bangun ulang indeks (ukur waktu build)
    POST /api/kb/{koleksi}/unload       lepas snapshot indeks dari RAM (mis. setelah uji 10^6 chunk)

Jalankan:  python -m server   (atau: uvicorn server.app:app)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag.config import load_config
from rag.service import RAGService

log = logging.getLogger("server")
CFG = load_config()
svc: RAGService | None = None
_startup_info: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc
    # 100+ pengguna konkuren menunggu di semaphore pipeline -> perlu thread pool besar
    anyio.to_thread.current_default_thread_limiter().total_tokens = 512
    svc = RAGService(CFG, start_sampler=True)
    if (CFG.get("server") or {}).get("warmup", True):
        t = time.perf_counter()
        _startup_info["warmup_ms"] = await run_in_threadpool(svc.warmup, True)
        log.info("Warmup selesai dalam %.1fs: %s", time.perf_counter() - t, _startup_info["warmup_ms"])
    _startup_info["started_at"] = time.time()
    yield
    svc.close()


app = FastAPI(title="Asisten Informasi Akademik UGM — RAG API", version="0.1.0", lifespan=lifespan)


def S() -> RAGService:
    if svc is None:
        raise HTTPException(503, "Server belum siap")
    return svc


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection: str | None = None
    mode: str = "baseline"
    stream: bool = True
    generate: bool = True  # False = retrieval-only (untuk uji beban tahap retrieval)
    client: str | None = None


class CollectionRequest(BaseModel):
    name: str
    description: str = ""
    synthetic: bool = False


class UrlRequest(BaseModel):
    url: str
    title: str | None = None
    category: str | None = None
    rebuild: bool = True


class RebuildRequest(BaseModel):
    full: bool = False
    mode: str = "baseline"


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, (ValueError, FileExistsError)):
        return HTTPException(400, str(exc))
    return HTTPException(500, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ status
@app.get("/api/health")
def health():
    s = S()
    return {
        "status": "ok",
        "llm": s.llm.health(),
        "llm_name": s.llm.describe(),
        "embedder": s.cfg["models"]["embedder"]["name"],
        "reranker": s.cfg["models"]["reranker"]["name"] if s.reranker_enabled else None,
        "default_collection": s.default_collection,
        "startup": _startup_info,
        "hpc": _hpc_info(),
    }


_HPC_INFO: dict[str, Any] = {}


def _hpc_info() -> dict[str, Any]:
    """Status modul C++ rag_hpc (CUDA/OpenMP) — di-cache, cukup diprobe sekali."""
    from rag.backends import HPC_STATUS

    if not _HPC_INFO:
        _HPC_INFO.update(HPC_STATUS)
        if HPC_STATUS.get("loaded"):
            try:
                from rag import hpc

                _HPC_INFO.update(hpc.info())
            except Exception as exc:  # noqa: BLE001
                _HPC_INFO["error"] = str(exc)
    return _HPC_INFO


@app.get("/api/modes")
def modes():
    return S().mode_status()


@app.get("/api/metrics")
def metrics(window: float = 60.0, mode: str | None = None):
    s = S()
    out = s.metrics.snapshot(window, mode)
    if mode is None:
        out["by_mode"] = {m: s.metrics.snapshot(window, m) for m in s.modes}
    if s.sampler:
        out["resources"] = s.sampler.latest()
        out["resources_summary"] = s.sampler.summary(window)
    return out


# ------------------------------------------------------------------ chat
_DONE = object()


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    s = S()
    cancel = threading.Event()
    events = s.answer_stream(
        req.question, collection=req.collection, mode=req.mode, generate=req.generate, client=req.client,
        cancel=cancel,
    )
    if req.stream:
        async def watch_disconnect():
            # Request yang masih antre tidak menulis apa pun ke socket, jadi putusnya klien harus dipantau aktif.
            while not cancel.is_set():
                if await request.is_disconnected():
                    cancel.set()
                    return
                await asyncio.sleep(0.25)

        async def ndjson():
            watcher = asyncio.create_task(watch_disconnect())
            try:
                while True:
                    # abandon_on_cancel: bila klien pergi, coroutine tidak ikut menunggu thread yang sedang antre.
                    ev = await anyio.to_thread.run_sync(next, events, _DONE, abandon_on_cancel=True)
                    if ev is _DONE:
                        break
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
            finally:
                cancel.set()
                watcher.cancel()
                try:
                    events.close()
                except ValueError:
                    pass  # generator masih berjalan di thread lain; ia berhenti sendiri lewat flag cancel

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    out: dict[str, Any] = {}
    for ev in events:
        if ev["type"] == "meta":
            out.update({k: v for k, v in ev.items() if k not in ("type", "timings_ms")})
        elif ev["type"] == "done":
            out.update({k: v for k, v in ev.items() if k != "type"})
        elif ev["type"] == "error":
            return JSONResponse({"error": ev["message"], "request_id": ev.get("request_id")}, status_code=400)
    return out


# ------------------------------------------------------------------ koleksi
@app.get("/api/collections")
def list_collections():
    return S().collections()


@app.post("/api/collections")
def create_collection(req: CollectionRequest):
    try:
        return S().create_collection(req.name, req.description, req.synthetic)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc


@app.get("/api/kb/{collection}/documents")
def list_documents(collection: str):
    try:
        kb = S().kb(collection)
        return {"stats": kb.stats(), "documents": kb.list_documents()}
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc


@app.get("/api/kb/{collection}/documents/{doc_id}/chunks")
def document_chunks(collection: str, doc_id: int, limit: int = 50, offset: int = 0):
    kb = S().kb(collection)
    doc = kb.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Dokumen tidak ditemukan")
    return {"document": doc, "chunks": kb.doc_chunks(doc_id, limit=limit, offset=offset)}


@app.post("/api/kb/{collection}/documents")
def upload_documents(
    collection: str,
    files: list[UploadFile] = File(...),
    title: str | None = Form(None),
    category: str | None = Form(None),
    source_url: str | None = Form(None),
    rebuild: bool = Form(True),
):
    s = S()
    try:
        kb = s.kb(collection)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc
    results = []
    for f in files:
        data = f.file.read()
        try:
            res = s.add_file(
                kb.name, data, f.filename or "dokumen",
                title=(title or None) if len(files) == 1 else None,
                category=category or None,
                source_url=(source_url or None) if len(files) == 1 else None,
                content_type=f.content_type,
            )
        except Exception as exc:  # noqa: BLE001
            res = {"status": "error", "filename": f.filename, "error": str(exc)}
        results.append(res)
    index = None
    if rebuild and any(r.get("status") == "ok" for r in results):
        index = s.rebuild(kb.name)
    return {"results": results, "index": index}


@app.post("/api/kb/{collection}/url")
def add_url(collection: str, req: UrlRequest):
    s = S()
    try:
        res = s.add_url(collection, req.url, title=req.title or None, category=req.category or None)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc
    index = s.rebuild(collection) if req.rebuild and res.get("status") == "ok" else None
    return {"result": res, "index": index}


@app.delete("/api/kb/{collection}/documents/{doc_id}")
def delete_document(collection: str, doc_id: int, rebuild: bool = True):
    s = S()
    try:
        if s.kb(collection).get_document(doc_id) is None:
            raise HTTPException(404, "Dokumen tidak ditemukan")
    except FileNotFoundError as exc:
        raise _http_error(exc) from exc
    n = s.delete_document(collection, doc_id)
    index = s.rebuild(collection) if rebuild else None
    return {"deleted_chunks": n, "index": index}


@app.post("/api/kb/{collection}/rebuild")
def rebuild(collection: str, req: RebuildRequest):
    try:
        return S().rebuild(collection, full=req.full, mode=req.mode)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc) from exc


@app.post("/api/kb/{collection}/unload")
def unload(collection: str):
    """Lepas snapshot indeks koleksi dari RAM (dimuat ulang otomatis saat dipakai lagi)."""
    return S().unload(collection)
