"""RAGService: orkestrasi model, knowledge base, indeks, dan pipeline tanya-jawab.

Alur online (per pertanyaan), setiap tahap diukur waktunya:

    queue_retrieval -> embed_query -> search_vector -> search_bm25 -> fusion
    -> fetch_chunks -> rerank -> [filter relevansi] -> queue_llm -> llm_prefill
    -> llm_decode

Jika tidak ada chunk yang lolos ambang relevansi, LLM tidak dipanggil dan
jawaban penolakan baku dikembalikan — jawaban tidak pernah dibuat tanpa
dasar dokumen di knowledge base.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import re
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, Iterator

from .backends import BM25_BACKENDS, HPC_STATUS, VECTOR_BACKENDS
from .batching import BatchRetriever
from .config import ModeConfig, chunk_params, load_config, mode_configs, resolve_device
from .index.hybrid import rrf_fuse
from .indexer import IndexSnapshot, build_index, index_version, load_index
from .ingest import ingest_bytes
from .metrics import MetricsTracker, RequestLogger, ResourceSampler
from .prompts import build_messages
from .store import KnowledgeBase, valid_collection_name
from .text import tokenize
from .timing import StageTimer

log = logging.getLogger(__name__)
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001
        return -1.0


def _malloc_trim() -> None:
    """Kembalikan memori bebas glibc ke OS (Linux) agar RSS benar-benar turun setelah unload."""
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


class RequestCancelled(Exception):
    """Klien memutus koneksi sebelum request selesai (mis. akhir run uji beban)."""


class FifoSemaphore:
    """Semaphore adil (FIFO) yang bisa dibatalkan.

    Antrean baseline harus first-come-first-served agar p50/p95 mencerminkan
    antrean sebenarnya. `threading.Semaphore` tidak menjamin urutan, apalagi bila
    ditunggu dengan timeout berulang (penunggu lama bisa disalip pendatang baru,
    sehingga ekor latensi membengkak). Slot yang dilepas diserahkan langsung ke
    penunggu terdepan.
    """

    def __init__(self, value: int) -> None:
        self._free = max(1, int(value))
        self._lock = threading.Lock()
        self._waiters: deque[threading.Event] = deque()

    def acquire(self, cancel: threading.Event | None = None) -> None:
        if cancel is not None and cancel.is_set():
            raise RequestCancelled()
        with self._lock:
            if self._free > 0 and not self._waiters:
                self._free -= 1
                return
            ticket = threading.Event()
            self._waiters.append(ticket)
        while not ticket.wait(0.2):
            if cancel is not None and cancel.is_set():
                with self._lock:
                    if not ticket.is_set():
                        self._waiters.remove(ticket)
                        raise RequestCancelled()
                # slot sudah telanjur diserahkan ke kita -> kembalikan
                self.release()
                raise RequestCancelled()

    def release(self) -> None:
        with self._lock:
            if self._waiters:
                self._waiters.popleft().set()
            else:
                self._free += 1

    def queued(self) -> int:
        return len(self._waiters)


class RetrievalPool:
    """Antrean FIFO + worker tetap untuk tahap retrieval (slot = `pipeline_max_concurrency`).

    Retrieval (embedding kueri, pencarian vektor/BM25, fetch SQLite, rerank) sengaja
    tidak dijalankan di thread request: server memakai ratusan thread, dan setiap
    thread yang menyentuh NumPy/SQLite membawa arena malloc + koneksi + cache
    sendiri sehingga RSS membengkak mengikuti jumlah pengguna (sempat memicu OOM
    pada uji beban 10^6 chunk). Dengan worker tetap, memori dibatasi jumlah slot.
    """

    def __init__(self, name: str, workers: int) -> None:
        self._exec = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix=f"retrieval-{name}")

    def run(self, fn: Callable[[], Any], timer: StageTimer, cancel: threading.Event | None = None) -> Any:
        t_submit = time.perf_counter()

        def task() -> Any:
            timer.add("queue_retrieval", (time.perf_counter() - t_submit) * 1000)
            if cancel is not None and cancel.is_set():
                raise RequestCancelled()
            return fn()

        fut = self._exec.submit(task)
        while True:
            try:
                return fut.result(timeout=0.2)
            except FutureTimeout:
                if cancel is not None and cancel.is_set() and fut.cancel():
                    raise RequestCancelled() from None

    def close(self) -> None:
        self._exec.shutdown(wait=False, cancel_futures=True)


class RAGService:
    def __init__(self, cfg: dict | None = None, *, start_sampler: bool = False) -> None:
        self.cfg = cfg or load_config()
        self.kb_root = Path(self.cfg["paths"]["kb_dir"])
        self.kb_root.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(self.cfg["paths"]["results_dir"])
        self.modes = mode_configs(self.cfg)
        self.chunk_cfg = chunk_params(self.cfg)
        self.retr = self.cfg.get("retrieval") or {}
        self.refuse_message = " ".join(str((self.cfg.get("grounding") or {}).get("refuse_message", "")).split())
        self.default_collection = (self.cfg.get("kb") or {}).get("default_collection", "ugm")

        self.metrics = MetricsTracker()
        self.sampler = ResourceSampler() if start_sampler else None
        if self.sampler:
            self.sampler.start()
        server_cfg = self.cfg.get("server") or {}
        self.request_logger = RequestLogger(self.results_dir / "requests") if server_cfg.get("request_log") else None

        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._kbs: dict[str, KnowledgeBase] = {}
        self._snaps: dict[str, IndexSnapshot] = {}
        self._embedders: dict[str, Any] = {}
        self._reranker = None
        self._llm = None
        self._cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-stage")  # overlap CPU-GPU
        self._pools = {
            name: (
                BatchRetriever(self, m) if m.retrieval_batch_max > 1 else RetrievalPool(name, m.pipeline_max_concurrency),
                FifoSemaphore(m.llm_max_concurrency),
            )
            for name, m in self.modes.items()
        }
        if not (self.kb_root / self.default_collection / "collection.json").exists():
            KnowledgeBase.create(self.kb_root, self.default_collection, "Dokumen akademik resmi UGM", synthetic=False)

    # ================================================================ model
    def embedder(self, dtype: str = "float32"):
        """Satu instans embedder FP32 untuk semua mode; mode teroptimasi memakai FP16 lewat
        autocast (``encode(fp16=True)``) sehingga bobot tidak diduplikasi di VRAM."""
        with self._lock:
            if "float32" not in self._embedders:
                from .models.embedder import Embedder

                e = self.cfg["models"]["embedder"]
                self._embedders["float32"] = Embedder(
                    e["name"],
                    device=resolve_device(e.get("device")),
                    dtype="float32",
                    max_length=int(e.get("max_length", 512)),
                    pooling=e.get("pooling", "mean"),
                    query_prefix=e.get("query_prefix", ""),
                    passage_prefix=e.get("passage_prefix", ""),
                )
            return self._embedders["float32"]

    @property
    def reranker_enabled(self) -> bool:
        return bool((self.cfg["models"].get("reranker") or {}).get("enabled", False))

    @property
    def reranker(self):
        with self._lock:
            if self._reranker is None and self.reranker_enabled:
                from .models.reranker import Reranker

                r = self.cfg["models"]["reranker"]
                self._reranker = Reranker(
                    r["name"],
                    device=resolve_device(r.get("device")),
                    max_length=int(r.get("max_length", 512)),
                    batch_size=int(r.get("batch_size", 16)),
                )
            return self._reranker

    @property
    def llm(self):
        with self._lock:
            if self._llm is None:
                from .models.llm import create_llm

                self._llm = create_llm(self.cfg["models"]["llm"])
            return self._llm

    def warmup(self, with_llm: bool = True) -> dict[str, float]:
        out: dict[str, float] = {}
        t = time.perf_counter()
        emb = self.embedder()
        emb.encode_query("pemanasan model")
        emb.encode(["pemanasan model fp16", "batch"], batch_size=2, is_query=True, fp16=True)
        out["embedder_ms"] = (time.perf_counter() - t) * 1000
        if self.reranker_enabled:
            t = time.perf_counter()
            self.reranker.score("pemanasan", ["model reranker"])
            self.reranker.score_pairs([("pemanasan", "model reranker fp16")] * 2, fp16=True)
            out["reranker_ms"] = (time.perf_counter() - t) * 1000
        if "cuda_topk" in VECTOR_BACKENDS:
            import numpy as np

            t = time.perf_counter()
            try:  # kompilasi JIT PTX -> SASS (sekali per mesin, lalu di-cache driver)
                probe = np.eye(8, emb.dim, dtype=np.float32)
                VECTOR_BACKENDS["cuda_topk"](probe, dtype="float16").search_batch(probe[:2], 4)
                out["cuda_kernels_ms"] = (time.perf_counter() - t) * 1000
            except Exception as exc:  # noqa: BLE001
                log.warning("Warmup kernel CUDA gagal: %s", exc)
        if with_llm:
            from .models.llm import warmup

            try:
                out["llm_ms"] = warmup(self.llm)
            except Exception as exc:  # noqa: BLE001
                log.warning("Warmup LLM gagal: %s", exc)
        return out

    # ================================================================ mode
    def register_mode(self, m: ModeConfig) -> None:
        """Tambah mode saat runtime (dipakai tools/ablation.py untuk konfigurasi kumulatif)."""
        with self._lock:
            self.modes[m.name] = m
            self._pools[m.name] = (
                BatchRetriever(self, m) if m.retrieval_batch_max > 1 else RetrievalPool(m.name, m.pipeline_max_concurrency),
                FifoSemaphore(m.llm_max_concurrency),
            )

    def mode_status(self) -> dict[str, dict[str, Any]]:
        status = {}
        for name, m in self.modes.items():
            missing = []
            if m.vector_backend not in VECTOR_BACKENDS:
                missing.append(f"vector_backend={m.vector_backend}")
            if m.bm25_backend not in BM25_BACKENDS:
                missing.append(f"bm25_backend={m.bm25_backend}")
            status[name] = {
                "available": not missing,
                "missing": missing,
                "description": m.description,
                "params": {k: v for k, v in m.__dict__.items() if k not in ("name", "description")},
            }
        return {"modes": status, "hpc_extension": dict(HPC_STATUS)}

    def get_mode(self, name: str | None) -> ModeConfig:
        name = name or "baseline"
        if name not in self.modes:
            raise ValueError(f"Mode '{name}' tidak dikenal (tersedia: {list(self.modes)})")
        st = self.mode_status()["modes"][name]
        if not st["available"]:
            raise ValueError(f"Mode '{name}' belum tersedia: backend belum terpasang ({', '.join(st['missing'])})")
        return self.modes[name]

    # ================================================================ koleksi
    def kb(self, name: str | None = None) -> KnowledgeBase:
        name = name or self.default_collection
        with self._lock:
            if name not in self._kbs:
                self._kbs[name] = KnowledgeBase(self.kb_root, name)
            return self._kbs[name]

    def create_collection(self, name: str, description: str = "", synthetic: bool = False) -> dict[str, Any]:
        if not valid_collection_name(name):
            raise ValueError("Nama koleksi hanya boleh huruf kecil, angka, '-' dan '_'")
        kb = KnowledgeBase.create(self.kb_root, name, description, synthetic=synthetic)
        with self._lock:
            self._kbs[name] = kb
        return kb.stats()

    def collections(self) -> list[dict[str, Any]]:
        return [self.kb(n).stats() for n in KnowledgeBase.list_names(self.kb_root)]

    def snapshot(self, name: str | None = None) -> IndexSnapshot | None:
        kb = self.kb(name)
        ver = index_version(kb)
        with self._lock:
            snap = self._snaps.get(kb.name)
            if ver is None:
                return None
            if snap is None or snap.version != ver:
                snap = load_index(kb)
                if snap is not None:
                    self._snaps[kb.name] = snap
            return snap

    # ================================================================ ingest
    def rebuild(self, collection: str | None = None, *, full: bool = False, mode: str = "baseline", log_fn=None) -> dict:
        m = self.get_mode(mode)
        kb = self.kb(collection)
        with self._build_lock:
            snap, meta = build_index(
                kb, self.embedder(), batch_size=m.embed_batch_size, fp16=(m.embed_dtype == "float16"),
                full=full, mode=m.name, log_fn=log_fn,
            )
        with self._lock:
            self._snaps[kb.name] = snap
        return meta

    def add_file(
        self,
        collection: str | None,
        data: bytes,
        filename: str,
        *,
        title: str | None = None,
        category: str | None = None,
        source_url: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        kb = self.kb(collection)
        return ingest_bytes(
            kb, data, filename, chunk_cfg=self.chunk_cfg, title=title, category=category,
            source_url=source_url, content_type=content_type,
        )

    def add_url(self, collection: str | None, url: str, *, title: str | None = None, category: str | None = None) -> dict:
        from scraper.fetcher import PoliteFetcher

        res = PoliteFetcher().fetch(url)
        if not res.ok:
            raise ValueError(f"Gagal mengambil {url}: {res.error or res.status}")
        name = Path(res.final_url.split("?")[0]).name or "halaman.html"
        return self.add_file(
            collection, res.content, name, title=title, category=category, source_url=url,
            content_type=res.content_type,
        )

    def delete_document(self, collection: str | None, doc_id: int) -> int:
        return self.kb(collection).delete_document(doc_id)

    # ================================================================ retrieval
    def retrieve(
        self, question: str, collection: str | None, mode: ModeConfig, timer: StageTimer
    ) -> tuple[list[dict[str, Any]], IndexSnapshot | None]:
        kb = self.kb(collection)
        snap = self.snapshot(kb.name)
        if snap is None or snap.size == 0:
            return [], snap
        r = self.retr
        # model dimuat di luar tahap yang diukur agar waktu cold-start/unduh tidak mencemari profil
        embedder = self.embedder()
        reranker = self.reranker if self.reranker_enabled else None
        with timer.stage("embed_query"):
            q = embedder.encode([question], batch_size=1, is_query=True, fp16=mode.model_fp16)[0]
        with timer.stage("search_vector"):
            v_idx, v_scores = snap.vector_backend(mode.vector_backend, mode.vector_threads, mode.vector_dtype).search(
                q, int(r.get("top_k_vector", 50))
            )
        with timer.stage("search_bm25"):
            b_idx, b_scores = snap.bm25_backend(mode.bm25_backend, mode.bm25_threads).search(
                tokenize(question), int(r.get("top_k_bm25", 50))
            )
        with timer.stage("fusion"):
            fused, v_map, b_map = self._fuse(v_idx, v_scores, b_idx, b_scores)
        with timer.stage("fetch_chunks"):
            rows = kb.get_chunks([int(snap.chunk_ids[i]) for i, _ in fused])
        hits = self._build_hits(fused, v_map, b_map, rows, snap)
        if reranker is not None and hits:
            with timer.stage("rerank"):
                scores = reranker.score(
                    question, [f"{h['title']}\n{h['text']}" for h in hits],
                    fp16=mode.model_fp16, batch_size=mode.rerank_batch_size or None,
                )
            for h, s in zip(hits, scores):
                h["rerank_score"] = round(float(s), 4)
            hits.sort(key=lambda h: -h["rerank_score"])
        return hits, snap

    def retrieve_batch(
        self, questions: list[str], collection: str | None, mode: ModeConfig, timers: list[StageTimer]
    ) -> tuple[list[list[dict[str, Any]]], IndexSnapshot | None, dict[str, Any]]:
        """Retrieval untuk satu batch permintaan konkuren (dipanggil BatchRetriever).

        Setiap tahap dijalankan sekali untuk seluruh batch; durasi tahap dicatat ke timer
        setiap permintaan (semua permintaan dalam batch mengalami waktu yang sama).
        """
        kb = self.kb(collection)
        snap = self.snapshot(kb.name)
        n = len(questions)
        info: dict[str, Any] = {"batch_size": n}
        if snap is None or snap.size == 0:
            return [[] for _ in questions], snap, info
        r = self.retr
        embedder = self.embedder()
        reranker = self.reranker if self.reranker_enabled else None
        kv, kbm = int(r.get("top_k_vector", 50)), int(r.get("top_k_bm25", 50))
        vec = snap.vector_backend(mode.vector_backend, mode.vector_threads, mode.vector_dtype)
        bm = snap.bm25_backend(mode.bm25_backend, mode.bm25_threads)

        def add_all(stage: str, ms: float) -> None:
            for t in timers:
                t.add(stage, ms)

        t = time.perf_counter()
        Q = embedder.encode(questions, batch_size=n, is_query=True, fp16=mode.model_fp16, sort_by_length=False, overlap=False)
        add_all("embed_query", (time.perf_counter() - t) * 1000)

        token_lists = [tokenize(q) for q in questions]

        def run_vec():
            t0 = time.perf_counter()
            if getattr(vec, "supports_batch", False):
                res, kt = vec.search_batch(Q, kv)
            else:
                res, kt = [vec.search(Q[i], kv) for i in range(n)], {}
            return res, (time.perf_counter() - t0) * 1000, kt

        def run_bm25():
            t0 = time.perf_counter()
            if getattr(bm, "supports_batch", False):
                res, kt = bm.search_batch(token_lists, kbm)
            else:
                res, kt = [bm.search(tl, kbm) for tl in token_lists], {}
            return res, (time.perf_counter() - t0) * 1000, kt

        t = time.perf_counter()
        if mode.overlap_cpu_gpu:
            fut = self._cpu_pool.submit(run_bm25)  # BM25 di CPU selagi GPU menghitung similarity
            v_res, v_ms, v_kt = run_vec()
            b_res, b_ms, b_kt = fut.result()
        else:
            v_res, v_ms, v_kt = run_vec()
            b_res, b_ms, b_kt = run_bm25()
        wall = (time.perf_counter() - t) * 1000
        add_all("search_vector", v_ms)
        add_all("search_bm25", b_ms)
        info["search_wall_ms"] = round(wall, 3)
        info["overlap_saved_ms"] = round(max(0.0, v_ms + b_ms - wall), 3)
        info["kernel_ms"] = {**{f"vec_{k}": round(v, 3) for k, v in v_kt.items()},
                             **{f"bm25_{k}": round(v, 3) for k, v in b_kt.items()}}

        t = time.perf_counter()
        fused_all = [self._fuse(vi, vs, bi, bs) for (vi, vs), (bi, bs) in zip(v_res, b_res)]
        add_all("fusion", (time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        want = sorted({int(snap.chunk_ids[i]) for fused, _, _ in fused_all for i, _ in fused})
        rows = kb.get_chunks(want)
        add_all("fetch_chunks", (time.perf_counter() - t) * 1000)
        hits_all = [self._build_hits(fused, vm, bmap, rows, snap) for fused, vm, bmap in fused_all]

        if reranker is not None:
            pairs, owner = [], []
            for qi, hits in enumerate(hits_all):
                for h in hits:
                    pairs.append((questions[qi], f"{h['title']}\n{h['text']}"))
                    owner.append(h)
            t = time.perf_counter()
            scores = reranker.score_pairs(
                pairs, fp16=mode.model_fp16, batch_size=mode.rerank_batch_size or None, sort_by_length=True
            )
            add_all("rerank", (time.perf_counter() - t) * 1000)
            info["rerank_pairs"] = len(pairs)
            for h, s in zip(owner, scores):
                h["rerank_score"] = round(float(s), 4)
            for hits in hits_all:
                hits.sort(key=lambda h: -h["rerank_score"])
        return hits_all, snap, info

    def _fuse(self, v_idx, v_scores, b_idx, b_scores):
        r = self.retr
        fused = rrf_fuse([v_idx, b_idx], k=int(r.get("rrf_k", 60)), limit=int(r.get("rerank_candidates", 20)))
        v_map = {int(i): (rank, float(s)) for rank, (i, s) in enumerate(zip(v_idx, v_scores))}
        b_map = {int(i): (rank, float(s)) for rank, (i, s) in enumerate(zip(b_idx, b_scores))}
        return fused, v_map, b_map

    @staticmethod
    def _build_hits(fused, v_map, b_map, rows, snap) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for i, rrf in fused:
            cid = int(snap.chunk_ids[i])
            row = rows.get(cid)
            if row is None:  # chunk sudah dihapus tapi indeks belum dibangun ulang
                continue
            vr = v_map.get(i)
            br = b_map.get(i)
            hits.append(
                {
                    **row,
                    "chunk_id": cid,
                    "rrf_score": round(rrf, 5),
                    "vector_score": round(vr[1], 4) if vr else None,
                    "vector_rank": vr[0] + 1 if vr else None,
                    "bm25_score": round(br[1], 3) if br else None,
                    "bm25_rank": br[0] + 1 if br else None,
                    "rerank_score": None,
                }
            )
        return hits

    def _select_context(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        k = int(self.retr.get("top_k_context", 5))
        if self.reranker_enabled:
            thr = float(self.retr.get("min_rerank_score", 0.1))
            keep = [h for h in hits if (h["rerank_score"] or 0) >= thr]
        else:
            thr = float(self.retr.get("min_vector_score", 0.8))
            keep = [h for h in hits if (h["vector_score"] or 0) >= thr]
        return keep[:k]

    @staticmethod
    def _source_view(n: int, h: dict[str, Any]) -> dict[str, Any]:
        return {
            "n": n,
            "chunk_id": h["chunk_id"],
            "doc_id": h["doc_id"],
            "title": h["title"],
            "url": h.get("source_url"),
            "category": h.get("category"),
            "page_start": h.get("page_start"),
            "page_end": h.get("page_end"),
            "rerank_score": h.get("rerank_score"),
            "vector_score": h.get("vector_score"),
            "bm25_score": h.get("bm25_score"),
            "text": h["text"],
        }

    # ================================================================ jawab
    def answer_stream(
        self,
        question: str,
        *,
        collection: str | None = None,
        mode: str | None = "baseline",
        generate: bool = True,
        client: str | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generator event: meta -> token* -> done (atau error).

        `cancel` (opsional) diset oleh server saat klien memutus koneksi: request yang
        masih antre/berjalan dihentikan supaya tidak memakan slot pipeline/LLM.
        """
        timer = StageTimer()
        req_id = uuid.uuid4().hex[:12]
        collection = collection or self.default_collection
        rec: dict[str, Any] = {
            "request_id": req_id, "collection": collection, "mode": mode, "generate": generate,
            "client": client, "question_chars": len(question),
        }
        self.metrics.start(mode)
        ttft_ms: float | None = None
        answer_parts: list[str] = []
        try:
            question = (question or "").strip()
            if not question:
                raise ValueError("Pertanyaan kosong")
            m = self.get_mode(mode)
            kb = self.kb(collection)
            retr_exec, sem_llm = self._pools[m.name]
            if isinstance(retr_exec, BatchRetriever):
                hits, snap, batch_info = retr_exec.run(question, collection, timer, cancel)
            else:
                hits, snap = retr_exec.run(lambda: self.retrieve(question, collection, m, timer), timer, cancel)
                batch_info = {}
            if snap is None or snap.size == 0:
                raise ValueError(
                    f"Knowledge base '{collection}' masih kosong atau belum diindeks. Tambahkan dokumen terlebih dahulu."
                )
            context = self._select_context(hits)
            sources = [self._source_view(i, h) for i, h in enumerate(context, 1)]
            yield {
                "type": "meta",
                "request_id": req_id,
                "collection": collection,
                "synthetic": kb.synthetic or bool(snap.meta.get("fake_embeddings")),
                "mode": m.name,
                "sources": sources,
                "n_candidates": len(hits),
                "top_candidate_score": hits[0]["rerank_score"] if hits else None,
                "timings_ms": timer.as_dict(),
                "batch": batch_info,
            }

            refused = False
            llm_stats: dict[str, Any] = {}
            if not context:
                refused = True
                ttft_ms = timer.elapsed_ms()
                answer_parts.append(self.refuse_message)
                yield {"type": "token", "text": self.refuse_message}
            elif generate:
                messages = build_messages(question, context, self.refuse_message)
                t_q = time.perf_counter()
                sem_llm.acquire(cancel)
                timer.add("queue_llm", (time.perf_counter() - t_q) * 1000)
                stream = None
                try:
                    t_llm = time.perf_counter()
                    t_first = None
                    stream = self.llm.stream_chat(messages)
                    for ev in stream:
                        if cancel is not None and cancel.is_set():
                            raise RequestCancelled()
                        if ev["type"] == "token":
                            if t_first is None:
                                t_first = time.perf_counter()
                                timer.add("llm_prefill", (t_first - t_llm) * 1000)
                                ttft_ms = timer.elapsed_ms()
                            answer_parts.append(ev["text"])
                            yield ev
                        elif ev["type"] == "stats":
                            llm_stats = {k: v for k, v in ev.items() if k != "type"}
                    if t_first is None:
                        timer.add("llm_prefill", (time.perf_counter() - t_llm) * 1000)
                    else:
                        timer.add("llm_decode", (time.perf_counter() - t_first) * 1000)
                finally:
                    if stream is not None and hasattr(stream, "close"):
                        stream.close()  # tutup koneksi ke backend -> generasi di server LLM ikut berhenti
                    sem_llm.release()

            answer = "".join(answer_parts).strip()
            if generate and context and self._is_refusal(answer):
                refused = True
            cited = sorted({int(n) for n in _CITE_RE.findall(answer) if 1 <= int(n) <= len(context)})
            e2e = timer.elapsed_ms()
            rec.update(
                stages=timer.as_dict(), e2e_ms=round(e2e, 3),
                ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
                refused=refused, n_context=len(context), cited=cited, llm=llm_stats,
            )
            yield {
                "type": "done",
                "request_id": req_id,
                "answer": answer,
                "refused": refused,
                "cited": cited,
                "uncited_warning": bool(generate and context and not refused and not cited),
                "timings_ms": timer.as_dict(),
                "e2e_ms": round(e2e, 3),
                "ttft_ms": rec["ttft_ms"],
                "llm": llm_stats,
            }
        except RequestCancelled:
            log.info("request %s dibatalkan (klien terputus) setelah %.0f ms", req_id, timer.elapsed_ms())
            rec.update(error="cancelled", stages=timer.as_dict(), e2e_ms=round(timer.elapsed_ms(), 3))
        except Exception as exc:  # noqa: BLE001
            log.exception("request %s gagal", req_id)
            rec.update(error=str(exc), stages=timer.as_dict(), e2e_ms=round(timer.elapsed_ms(), 3))
            yield {"type": "error", "request_id": req_id, "message": str(exc)}
        finally:
            if "e2e_ms" not in rec:  # klien memutus koneksi di tengah stream
                rec.update(error="cancelled", stages=timer.as_dict(), e2e_ms=round(timer.elapsed_ms(), 3))
            self.metrics.finish(rec)
            if self.request_logger:
                self.request_logger.write({**rec, "ts": time.time()})

    def _is_refusal(self, answer: str) -> bool:
        a = " ".join(answer.split()).lower()
        ref = self.refuse_message.lower()
        return bool(a) and (a.startswith(ref[:60]) or a in ref)

    def answer(self, question: str, **kw) -> dict[str, Any]:
        """Versi non-streaming: kumpulkan seluruh event menjadi satu dict."""
        out: dict[str, Any] = {}
        for ev in self.answer_stream(question, **kw):
            if ev["type"] == "meta":
                out.update(sources=ev["sources"], synthetic=ev["synthetic"], retrieval_timings_ms=ev["timings_ms"])
            elif ev["type"] == "done":
                out.update({k: v for k, v in ev.items() if k != "type"})
            elif ev["type"] == "error":
                out.update(error=ev["message"])
        return out

    def unload(self, name: str | None = None) -> dict[str, Any]:
        """Lepas snapshot indeks dari memori (semua bila `name` None); dimuat ulang otomatis saat dipakai lagi."""
        rss_before = _rss_mb()
        with self._lock:
            names = [name] if name else list(self._snaps)
            dropped = [n for n in names if self._snaps.pop(n, None) is not None]
        gc.collect()
        _malloc_trim()
        vram = None
        try:  # kembalikan blok VRAM yang di-cache alokator PyTorch (aktivasi batch rerank, dll.)
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram = round(torch.cuda.memory_reserved() / 2**20, 1)
        except Exception:  # noqa: BLE001
            pass
        return {"unloaded": dropped, "rss_mb_before": rss_before, "rss_mb_after": _rss_mb(), "torch_reserved_mb": vram}

    def close(self) -> None:
        if self.sampler:
            self.sampler.stop()
        for pool, _ in self._pools.values():
            pool.close()
        self._cpu_pool.shutdown(wait=False)

