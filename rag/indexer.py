"""Pembangunan & pemuatan indeks (embedding + BM25) untuk satu koleksi."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backends import BM25_BACKENDS, VECTOR_BACKENDS
from .index.bm25 import BM25Index
from .store import KnowledgeBase, now_iso
from .text import tokenize
from .timing import StageTimer

log = logging.getLogger(__name__)
INDEX_FORMAT = 1


@dataclass
class IndexSnapshot:
    """Indeks immutable yang sedang dipakai melayani kueri.

    Rebuild membuat snapshot baru lalu ditukar secara atomik, sehingga request
    yang sedang berjalan tetap memakai snapshot lama.
    """

    collection: str
    chunk_ids: np.ndarray
    embeddings: np.ndarray
    bm25: BM25Index
    meta: dict[str, Any]
    version: float
    _cache: dict = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def size(self) -> int:
        return int(self.chunk_ids.shape[0])

    def vector_backend(self, name: str, threads: int = 1, dtype: str = "float32"):
        key = ("vec", name, threads, dtype)
        with self._lock:
            if key not in self._cache:
                if name not in VECTOR_BACKENDS:
                    raise KeyError(f"backend vektor '{name}' tidak tersedia")
                self._cache[key] = VECTOR_BACKENDS[name](self.embeddings, threads=threads, dtype=dtype)
            return self._cache[key]

    def bm25_backend(self, name: str, threads: int = 1):
        key = ("bm25", name, threads)
        with self._lock:
            if key not in self._cache:
                if name not in BM25_BACKENDS:
                    raise KeyError(f"backend BM25 '{name}' tidak tersedia")
                self._cache[key] = BM25_BACKENDS[name](self.bm25, threads=threads)
            return self._cache[key]


def _meta_path(kb: KnowledgeBase) -> Path:
    return kb.index_dir / "index_meta.json"


def index_version(kb: KnowledgeBase) -> float | None:
    p = _meta_path(kb)
    return p.stat().st_mtime if p.exists() else None


def load_index(kb: KnowledgeBase) -> IndexSnapshot | None:
    meta_p = _meta_path(kb)
    if not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    emb = np.load(kb.index_dir / "embeddings.npy")
    ids = np.load(kb.index_dir / "chunk_ids.npy")
    bm25 = BM25Index.load(kb.index_dir)
    return IndexSnapshot(kb.name, ids, emb, bm25, meta, meta_p.stat().st_mtime)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_index(
    kb: KnowledgeBase, chunk_ids: np.ndarray, embeddings: np.ndarray, bm25: BM25Index, meta: dict[str, Any]
) -> None:
    _atomic_save_npy(kb.index_dir / "embeddings.npy", np.ascontiguousarray(embeddings, dtype=np.float32))
    _atomic_save_npy(kb.index_dir / "chunk_ids.npy", np.asarray(chunk_ids, dtype=np.int64))
    bm25.save(kb.index_dir)
    meta = dict(meta)
    meta.update(
        format=INDEX_FORMAT,
        n_chunks=int(len(chunk_ids)),
        max_chunk_id=int(chunk_ids.max()) if len(chunk_ids) else 0,
        dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        bm25_vocab=len(bm25.vocab),
        bm25_nnz=bm25.nnz,
        built_at=now_iso(),
    )
    tmp = kb.index_dir / "index_meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _meta_path(kb))


def build_bm25(kb: KnowledgeBase, ids: np.ndarray) -> BM25Index:
    """Bangun BM25 dengan urutan baris persis sama dengan `ids`."""

    def token_stream():
        id_list = ids.tolist()
        for i in range(0, len(id_list), 900):
            part = id_list[i : i + 900]
            got: dict[int, str] = {}
            for pids, texts in kb.iter_index_texts(part):
                got.update(zip(pids, texts))
            for c in part:
                yield tokenize(got.get(c, ""))

    return BM25Index.build(token_stream())


def build_index(
    kb: KnowledgeBase,
    embedder,
    *,
    batch_size: int = 1,
    fp16: bool = False,
    full: bool = False,
    mode: str = "baseline",
    log_fn: Callable[[str], None] | None = None,
) -> tuple[IndexSnapshot, dict[str, Any]]:
    """Bangun indeks koleksi.

    Inkremental (default): hanya chunk baru yang di-embed, embedding chunk lama
    dipakai ulang. ``full=True`` menghitung ulang seluruh embedding — dipakai
    untuk mengukur waktu pembangunan indeks.
    BM25 selalu dibangun ulang dari seluruh chunk.
    """
    say = log_fn or log.info
    timer = StageTimer()
    with timer.stage("load_chunk_ids"):
        ids = kb.all_chunk_ids()

    old_meta = json.loads(_meta_path(kb).read_text(encoding="utf-8")) if _meta_path(kb).exists() else {}
    reuse = (
        not full
        and old_meta.get("embed_model") == embedder.name
        and int(old_meta.get("dim", -1)) == embedder.dim
        and (kb.index_dir / "embeddings.npy").exists()
    )
    emb = np.zeros((len(ids), embedder.dim), dtype=np.float32)
    have = np.zeros(len(ids), dtype=bool)
    if reuse:
        with timer.stage("load_cached_embeddings"):
            old_ids = np.load(kb.index_dir / "chunk_ids.npy")
            old_emb = np.load(kb.index_dir / "embeddings.npy")
            pos = {int(c): i for i, c in enumerate(old_ids)}
            for i, c in enumerate(ids):
                j = pos.get(int(c))
                if j is not None:
                    emb[i] = old_emb[j]
                    have[i] = True
            del old_emb
    missing = ids[~have]
    row_of = {int(c): i for i, c in enumerate(ids)}

    n_new = len(missing)
    say(f"[{kb.name}] {len(ids)} chunk, perlu di-embed: {n_new} (batch={batch_size}, fp16={fp16}, model={embedder.name})")
    with timer.stage("embed"):
        done = 0
        t_last = time.perf_counter()
        t_start = t_last
        for chunk_ids_part, texts in kb.iter_index_texts(missing.tolist() if reuse else None):
            vecs = embedder.encode(texts, batch_size=batch_size, is_query=False, fp16=fp16)
            for c, v in zip(chunk_ids_part, vecs):
                r = row_of.get(c)
                if r is not None:  # chunk yang masuk saat build berjalan diabaikan
                    emb[r] = v
            done += len(texts)
            now = time.perf_counter()
            if now - t_last > 5:
                rate = done / (now - t_start)
                say(f"  embed {done}/{n_new} ({rate:.0f} chunk/s)")
                t_last = now

    with timer.stage("bm25"):
        bm25 = build_bm25(kb, ids)

    meta = {
        "collection": kb.name,
        "embed_model": embedder.name,
        "embed_device": embedder.device,
        "embed_dtype": "float16" if fp16 else embedder.dtype_name,
        "embed_batch_size": batch_size,
        "mode": mode,
        "full_rebuild": bool(full or not reuse),
        "n_embedded": int(n_new),
        "synthetic": kb.synthetic,
        "fake_embeddings": False,
    }
    with timer.stage("save"):
        save_index(kb, ids, emb, bm25, {**meta, "timings_ms": {}})
    timings = timer.as_dict()
    timings["total"] = round(sum(timings.values()), 3)
    meta["timings_ms"] = timings
    # tulis ulang meta dengan timing final
    meta_full = json.loads(_meta_path(kb).read_text(encoding="utf-8"))
    meta_full["timings_ms"] = timings
    _meta_path(kb).write_text(json.dumps(meta_full, indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"[{kb.name}] indeks selesai: " + ", ".join(f"{k}={v:.0f}ms" for k, v in timings.items()))
    snap = IndexSnapshot(kb.name, ids, emb, bm25, meta_full, _meta_path(kb).stat().st_mtime)
    return snap, meta_full
