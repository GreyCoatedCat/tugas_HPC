"""Ingest paralel (pool proses + pecah PDF tebal per rentang halaman) == ingest sekuensial."""

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from rag.ingest import PAGES_PER_TASK, ingest_paths  # noqa: E402
from rag.store import KnowledgeBase  # noqa: E402

CHUNK_CFG = {"chunk_words": 120, "overlap_words": 20, "min_words": 10}
WORDS = ("mahasiswa wajib mengisi kartu rencana studi sebelum batas waktu yang ditetapkan "
         "fakultas dan program studi sesuai kalender akademik universitas").split()


def _make_pdf(path: Path, pages: int, title: str) -> None:
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        lines = [f"BAB {p // 10 + 1}", f"Pasal {p + 1}"]
        for ln in range(18):
            lines.append(" ".join(WORDS[(p + ln + i) % len(WORDS)] for i in range(10)) + f" halaman {p + 1}.")
        page.insert_text((60, 60), "\n".join(lines), fontsize=10)
    doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


def _snapshot(kb: KnowledgeBase) -> dict[str, list[str]]:
    return {d["title"]: [c["text"] for c in kb.doc_chunks(d["id"])] for d in kb.list_documents()}


def test_parallel_ingest_equals_sequential(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_pdf(src / "pendek.pdf", 3, "Dokumen Pendek")
    _make_pdf(src / "sedang.pdf", 12, "Dokumen Sedang")
    _make_pdf(src / "tebal.pdf", int(PAGES_PER_TASK * 2.5), "Buku Tebal")  # dipecah per rentang halaman
    (src / "catatan.txt").write_text("Pengumuman Registrasi\n\n" + " ".join(WORDS * 8), encoding="utf-8")

    snaps = {}
    for workers in (1, 4):
        kb = KnowledgeBase.create(tmp_path / "kb", f"bench-w{workers}", "uji kesetaraan ingest", synthetic=True)
        res = ingest_paths(kb, [src], chunk_cfg=CHUNK_CFG, workers=workers, log_fn=lambda *_: None)
        assert res["errors"] == [] and res["ok"] == 4
        snaps[workers] = _snapshot(kb)
    assert snaps[1] == snaps[4]
    assert len(snaps[4]["Buku Tebal"]) > 0
