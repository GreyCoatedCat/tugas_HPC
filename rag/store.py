"""Penyimpanan knowledge base: satu folder per koleksi.

    data/kb/<koleksi>/
        collection.json   metadata koleksi (deskripsi, synthetic?)
        kb.sqlite         tabel documents + chunks (teks bersih)
        files/            salinan file asli yang diunggah
        index/            embeddings.npy, chunk_ids.npy, bm25.npz, index_meta.json

Koleksi bertanda ``synthetic: true`` (data dummy untuk benchmark) selalu
terpisah dari koleksi resmi sehingga tidak pernah tercampur ke jawaban chatbot.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .text import TextChunk

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_key     TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    source_url  TEXT,
    source_type TEXT,
    category    TEXT,
    file_path   TEXT,
    sha256      TEXT,
    n_pages     INTEGER,
    n_chunks    INTEGER DEFAULT 0,
    n_chars     INTEGER,
    added_at    TEXT,
    meta        TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    page_start INTEGER,
    page_end   INTEGER,
    n_words    INTEGER,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def valid_collection_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


class KnowledgeBase:
    def __init__(self, root: str | Path, name: str) -> None:
        if not valid_collection_name(name):
            raise ValueError("Nama koleksi hanya boleh huruf kecil, angka, '-' dan '_' (maks 64)")
        self.name = name
        self.dir = Path(root) / name
        info_path = self.dir / "collection.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Koleksi '{name}' belum ada")
        self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        self.db_path = self.dir / "kb.sqlite"
        self.files_dir = self.dir / "files"
        self.index_dir = self.dir / "index"
        self.files_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._conn().executescript(SCHEMA)

    # ------------------------------------------------------------ koleksi
    @classmethod
    def create(
        cls, root: str | Path, name: str, description: str = "", synthetic: bool = False, exist_ok: bool = True
    ) -> "KnowledgeBase":
        if not valid_collection_name(name):
            raise ValueError("Nama koleksi hanya boleh huruf kecil, angka, '-' dan '_' (maks 64)")
        d = Path(root) / name
        info_path = d / "collection.json"
        if info_path.exists():
            if not exist_ok:
                raise FileExistsError(f"Koleksi '{name}' sudah ada")
        else:
            d.mkdir(parents=True, exist_ok=True)
            info = {"name": name, "description": description, "synthetic": bool(synthetic), "created_at": now_iso()}
            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        return cls(root, name)

    @staticmethod
    def list_names(root: str | Path) -> list[str]:
        r = Path(root)
        if not r.exists():
            return []
        return sorted(p.name for p in r.iterdir() if (p / "collection.json").exists())

    @staticmethod
    def drop(root: str | Path, name: str) -> None:
        if not valid_collection_name(name):
            raise ValueError("nama koleksi tidak valid")
        d = Path(root) / name
        if (d / "collection.json").exists():
            shutil.rmtree(d)

    @property
    def synthetic(self) -> bool:
        return bool(self.info.get("synthetic"))

    # ------------------------------------------------------------ koneksi
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------ tulis
    def add_document(
        self,
        *,
        doc_key: str,
        title: str,
        chunks: Sequence[TextChunk],
        source_url: str | None = None,
        source_type: str | None = None,
        category: str | None = None,
        file_path: str | None = None,
        sha256: str | None = None,
        n_pages: int | None = None,
        n_chars: int | None = None,
        meta: dict | None = None,
    ) -> int:
        """Tambah dokumen + chunk-nya. Dokumen dengan doc_key sama akan diganti."""
        with self._write_lock:
            conn = self._conn()
            with conn:
                conn.execute("DELETE FROM documents WHERE doc_key = ?", (doc_key,))
                cur = conn.execute(
                    "INSERT INTO documents (doc_key, title, source_url, source_type, category, file_path, sha256,"
                    " n_pages, n_chunks, n_chars, added_at, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        doc_key, title, source_url, source_type, category, file_path, sha256,
                        n_pages, len(chunks), n_chars, now_iso(), json.dumps(meta or {}, ensure_ascii=False),
                    ),
                )
                doc_id = int(cur.lastrowid)
                conn.executemany(
                    "INSERT INTO chunks (doc_id, ordinal, page_start, page_end, n_words, text) VALUES (?,?,?,?,?,?)",
                    [(doc_id, i, c.page_start, c.page_end, c.n_words, c.text) for i, c in enumerate(chunks)],
                )
        return doc_id

    def bulk_add_documents(self, docs: Iterable[dict[str, Any]], batch: int = 500) -> int:
        """Insert cepat banyak dokumen (dipakai generator data dummy). Return jumlah chunk."""
        total = 0
        buf: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal total
            conn = self._conn()
            with conn:
                for d in buf:
                    cur = conn.execute(
                        "INSERT INTO documents (doc_key, title, source_url, source_type, category, n_pages,"
                        " n_chunks, n_chars, added_at, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            d["doc_key"], d["title"], d.get("source_url"), d.get("source_type", "synthetic"),
                            d.get("category"), d.get("n_pages"), len(d["chunks"]), d.get("n_chars"), now_iso(),
                            json.dumps(d.get("meta") or {}, ensure_ascii=False),
                        ),
                    )
                    doc_id = int(cur.lastrowid)
                    conn.executemany(
                        "INSERT INTO chunks (doc_id, ordinal, page_start, page_end, n_words, text)"
                        " VALUES (?,?,?,?,?,?)",
                        [(doc_id, i, c.page_start, c.page_end, c.n_words, c.text) for i, c in enumerate(d["chunks"])],
                    )
                    total += len(d["chunks"])
            buf.clear()

        with self._write_lock:
            for d in docs:
                buf.append(d)
                if len(buf) >= batch:
                    flush()
            if buf:
                flush()
        return total

    def delete_document(self, doc_id: int) -> int:
        with self._write_lock:
            conn = self._conn()
            with conn:
                row = conn.execute("SELECT n_chunks, file_path FROM documents WHERE id = ?", (doc_id,)).fetchone()
                if row is None:
                    return 0
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        fp = row["file_path"]
        if fp:
            p = Path(fp)
            if p.exists() and self.files_dir in p.resolve().parents:
                p.unlink()
        return int(row["n_chunks"] or 0)

    # ------------------------------------------------------------ baca
    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT id, doc_key, title, source_url, source_type, category, n_pages, n_chunks, n_chars, added_at"
            " FROM documents ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> dict[str, Any] | None:
        r = self._conn().execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def find_by_sha(self, sha256: str) -> dict[str, Any] | None:
        r = self._conn().execute("SELECT id, title FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return dict(r) if r else None

    def doc_chunks(self, doc_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT id, ordinal, page_start, page_end, n_words, text FROM chunks WHERE doc_id = ?"
            " ORDER BY ordinal LIMIT ? OFFSET ?",
            (doc_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chunks(self, ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        ids = [int(i) for i in ids]
        conn = self._conn()
        for i in range(0, len(ids), 900):
            part = ids[i : i + 900]
            q = (
                "SELECT c.id, c.doc_id, c.ordinal, c.page_start, c.page_end, c.text, d.title, d.source_url,"
                " d.category, d.doc_key, d.source_type FROM chunks c JOIN documents d ON d.id = c.doc_id"
                f" WHERE c.id IN ({','.join('?' * len(part))})"
            )
            for r in conn.execute(q, part):
                out[int(r["id"])] = dict(r)
        return out

    def all_chunk_ids(self) -> np.ndarray:
        rows = self._conn().execute("SELECT id FROM chunks ORDER BY id").fetchall()
        return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))

    def iter_index_texts(self, ids: Sequence[int] | None = None, batch: int = 2048) -> Iterator[tuple[list[int], list[str]]]:
        """Iterasi (chunk_ids, teks_untuk_indeks) — teks = judul dokumen + isi chunk."""
        conn = self._conn()
        if ids is None:
            cur = conn.execute(
                "SELECT c.id, d.title, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
            )
            while True:
                rows = cur.fetchmany(batch)
                if not rows:
                    break
                yield [int(r[0]) for r in rows], [index_text(r[1], r[2]) for r in rows]
        else:
            ids = [int(i) for i in ids]
            for i in range(0, len(ids), 900):
                part = ids[i : i + 900]
                q = (
                    "SELECT c.id, d.title, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id"
                    f" WHERE c.id IN ({','.join('?' * len(part))}) ORDER BY c.id"
                )
                rows = conn.execute(q, part).fetchall()
                yield [int(r[0]) for r in rows], [index_text(r[1], r[2]) for r in rows]

    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks, max_id = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
        meta_path = self.index_dir / "index_meta.json"
        index_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        stale = index_meta is None or (
            int(index_meta.get("n_chunks", -1)) != int(n_chunks)
            or int(index_meta.get("max_chunk_id", -1)) != int(max_id)
        )
        return {
            "name": self.name,
            "description": self.info.get("description", ""),
            "synthetic": self.synthetic,
            "n_documents": int(n_docs),
            "n_chunks": int(n_chunks),
            "index": index_meta,
            "index_stale": stale and int(n_chunks) > 0,
        }


def index_text(title: str | None, text: str) -> str:
    return f"{title}\n{text}" if title else text
