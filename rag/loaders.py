"""Parsing dokumen (PDF/HTML/TXT/MD) menjadi teks bersih per halaman."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .cleaning import clean_pdf_pages, clean_plain_text, clean_wrapped_text, normalize_unicode

SUPPORTED_EXT = {".pdf": "pdf", ".html": "html", ".htm": "html", ".txt": "txt", ".md": "md"}


@dataclass
class ParsedDocument:
    title: str
    pages: list[tuple[int | None, str]]  # (nomor halaman | None, teks bersih)
    source_type: str
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(t for _, t in self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(t) for _, t in self.pages)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def humanize_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"^\d{6,}[-_ ]*(?:wpdm[_-])?", "", stem)
    return re.sub(r"[_\-]+", " ", stem).strip() or name


def _good_title(t: str | None) -> str | None:
    if not t:
        return None
    t = " ".join(t.split())
    t = re.sub(r"^Microsoft (?:Word|PowerPoint)\s*-\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\.(docx?|pdf|pptx?)$", "", t, flags=re.IGNORECASE)
    if not 6 <= len(t) <= 200 or re.fullmatch(r"[\W\d_]+", t):
        return None
    if t.lower() in {"untitled", "judul", "document", "dokumen", "title"}:
        return None
    return t


def detect_kind(filename: str | None, content_type: str | None, data: bytes) -> str | None:
    if data[:5] == b"%PDF-":
        return "pdf"
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return "pdf"
    if "html" in ct:
        return "html"
    if filename:
        kind = SUPPORTED_EXT.get(Path(filename.split("?")[0]).suffix.lower())
        if kind:
            return kind
    head = data[:2048].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")) or b"<body" in head:
        return "html"
    if "text/plain" in ct:
        return "txt"
    return None


def _pymupdf():
    try:
        import pymupdf
    except ImportError:  # versi lama PyMuPDF
        import fitz as pymupdf  # type: ignore
    return pymupdf


def pdf_info(source: bytes | str | Path) -> tuple[int, str | None]:
    """(jumlah halaman, judul metadata) tanpa mengekstrak teks."""
    pm = _pymupdf()
    doc = pm.open(stream=source, filetype="pdf") if isinstance(source, bytes) else pm.open(str(source))
    with doc:
        if doc.needs_pass:
            raise ValueError("PDF terenkripsi / butuh password")
        return len(doc), (doc.metadata or {}).get("title")


def extract_pdf_text(source: bytes | str | Path, start: int = 0, end: int | None = None) -> list[str]:
    """Ekstrak teks mentah halaman [start, end). Tahap paling mahal saat ingest PDF tebal,
    sehingga mode teroptimasi membagi rentang halaman ke beberapa proses."""
    pm = _pymupdf()
    doc = pm.open(stream=source, filetype="pdf") if isinstance(source, bytes) else pm.open(str(source))
    with doc:
        if doc.needs_pass:
            raise ValueError("PDF terenkripsi / butuh password")
        end = len(doc) if end is None else min(end, len(doc))
        return [doc[i].get_text("text", sort=True) for i in range(start, end)]


def pdf_document_from_raw(
    raw: list[str], meta_title: str | None, filename: str | None = None, title: str | None = None
) -> ParsedDocument:
    warnings: list[str] = []
    cleaned = clean_pdf_pages(raw)
    pages = [(i + 1, t) for i, t in enumerate(cleaned) if t.strip()]
    raw_chars = sum(len(r) for r in raw)
    if raw and raw_chars / len(raw) < 100:
        warnings.append(
            "Teks yang bisa diekstrak sangat sedikit — kemungkinan PDF hasil scan (OCR belum didukung)."
        )
    final_title = title or _good_title(meta_title) or humanize_filename(filename or "dokumen.pdf")
    return ParsedDocument(
        title=final_title,
        pages=pages,
        source_type="pdf",
        warnings=warnings,
        meta={"n_pages_total": len(raw), "raw_chars": raw_chars},
    )


def parse_pdf(data: bytes, filename: str | None = None, title: str | None = None) -> ParsedDocument:
    _, meta_title = pdf_info(data)
    return pdf_document_from_raw(extract_pdf_text(data), meta_title, filename, title)


def parse_html(data: bytes | str, url: str | None = None, title: str | None = None) -> ParsedDocument:
    html = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    warnings: list[str] = []
    text: str | None = None
    page_title: str | None = None
    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True, favor_recall=True
        )
        md = trafilatura.extract_metadata(html, default_url=url)
        page_title = getattr(md, "title", None) if md else None
    except Exception as exc:  # noqa: BLE001 - fallback ke BeautifulSoup
        warnings.append(f"trafilatura gagal: {exc}")
    if not text:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        text = root.get_text("\n")
        if not page_title and soup.title and soup.title.string:
            page_title = soup.title.string
        warnings.append("Konten utama diekstrak dengan fallback BeautifulSoup.")
    cleaned = clean_plain_text(text or "")
    if page_title:
        page_title = re.sub(r"\s*[|–\-]\s*(Universitas Gadjah Mada|UGM)\s*$", "", page_title.strip())
    final_title = title or _good_title(page_title) or (url or "Halaman web")
    return ParsedDocument(
        title=final_title,
        pages=[(None, cleaned)] if cleaned else [],
        source_type="html",
        warnings=warnings,
        meta={"raw_chars": len(html)},
    )


def parse_text(data: bytes, filename: str | None = None, title: str | None = None, kind: str = "txt") -> ParsedDocument:
    raw = data.decode("utf-8-sig", errors="replace")
    if kind == "md":
        body = normalize_unicode(raw)
        body = re.sub(r"^#{1,6}\s*", "", body, flags=re.MULTILINE)
        body = re.sub(r"[*_`]{1,3}([^*_`]+)[*_`]{1,3}", r"\1", body)
        cleaned = clean_plain_text(body)
    else:
        cleaned = clean_wrapped_text(raw)
    first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    first = re.sub(r"^#{1,6}\s*", "", first)
    first_title = first if 6 <= len(first) <= 200 and not first.endswith((".", ":", ";")) else None
    return ParsedDocument(
        title=title or first_title or humanize_filename(filename or "dokumen.txt"),
        pages=[(None, cleaned)] if cleaned.strip() else [],
        source_type=kind,
        meta={"raw_chars": len(raw)},
    )


def parse_bytes(
    data: bytes,
    filename: str | None = None,
    *,
    content_type: str | None = None,
    url: str | None = None,
    title: str | None = None,
) -> ParsedDocument:
    kind = detect_kind(filename, content_type, data)
    if kind == "pdf":
        return parse_pdf(data, filename, title)
    if kind == "html":
        return parse_html(data, url=url, title=title)
    if kind in ("txt", "md"):
        return parse_text(data, filename, title, kind)
    raise ValueError(
        f"Format tidak didukung: {filename or content_type or 'unknown'} (dukungan: PDF, HTML, TXT, MD)"
    )


def parse_file(path: str | Path, title: str | None = None) -> ParsedDocument:
    p = Path(path)
    return parse_bytes(p.read_bytes(), p.name, title=title)
