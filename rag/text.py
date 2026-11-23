"""Tokenisasi (untuk BM25) dan chunking teks dokumen."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

# Stopword Bahasa Indonesia (ringkas). Kata bermakna seperti "tidak" sengaja
# tetap dibuang karena BM25 hanya dipakai untuk mencocokkan kata kunci.
STOPWORDS = frozenset(
    """
    ada adalah agar akan aku anda apa apabila apakah atas atau bagaimana bagi bahwa
    banyak beberapa belum berapa bisa boleh dalam dan dapat dari demikian dengan di
    dia hal harus hanya ia ialah ini itu jika juga kah kami kamu kapan karena ke
    kepada ketika lah maka masih mau melalui mengapa menjadi mereka nya oleh pada
    para saat saja sama sampai saya se sebagai sebelum sedang sehingga sejak selama
    semua seperti serta setelah siapa suatu sudah supaya tanpa telah tentang
    terhadap tersebut tetapi tidak untuk yaitu yakni yang
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def normalize_for_index(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(normalize_for_index(text))
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(
    r"^(?:BAB\s+[IVXLCDM\d]+\b|Pasal\s+\d+[A-Za-z]?\b|Bagian\s+(?:Ke)?\w+\b|Paragraf\s+\d+\b)",
    re.IGNORECASE,
)
_SENT_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class TextChunk:
    text: str
    page_start: int | None
    page_end: int | None
    n_words: int


@dataclass
class _Unit:
    text: str
    page: int | None
    n: int
    heading: bool
    level: int = 0  # 1 = BAB / "A. JUDUL", 2 = judul huruf kapital, 3 = Pasal


_BAB_RE = re.compile(r"^BAB\s+[IVXLCDM\d]+\b", re.IGNORECASE)
_LETTER_SECTION_RE = re.compile(r"^[A-H]\.\s+\S")
_PASAL_ONLY_RE = re.compile(r"^Pasal\s+\d+[A-Za-z]?$")


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def heading_level(text: str) -> int:
    """Deteksi judul bagian untuk konteks chunk (0 = bukan judul)."""
    if len(text) <= 120 and _BAB_RE.match(text):
        return 1
    if len(text) <= 150 and _LETTER_SECTION_RE.match(text) and _upper_ratio(text) > 0.8:
        return 1
    if _PASAL_ONLY_RE.match(text):
        return 3
    if len(text) <= 80 and len(text.split()) >= 2 and not text.endswith((".", ";", ",", ":")) and _upper_ratio(text) > 0.85:
        return 2
    return 0


def _split_long(text: str, max_words: int) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for sent in _SENT_RE.split(text):
        words = sent.split()
        if not words:
            continue
        if len(words) > max_words:
            if buf:
                out.append(" ".join(buf))
                buf, n = [], 0
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i : i + max_words]))
            continue
        if buf and n + len(words) > max_words:
            out.append(" ".join(buf))
            buf, n = [], 0
        buf.append(sent)
        n += len(words)
    if buf:
        out.append(" ".join(buf))
    return out


def _units(pages: Iterable[tuple[int | None, str]], max_words: int) -> Iterator[_Unit]:
    for page, text in pages:
        for para in re.split(r"\n+", text or ""):
            para = " ".join(para.split())
            if not para:
                continue
            n = len(para.split())
            heading = bool(HEADING_RE.match(para))
            if n <= max_words:
                yield _Unit(para, page, n, heading, heading_level(para))
            else:
                for j, piece in enumerate(_split_long(para, max_words)):
                    yield _Unit(piece, page, len(piece.split()), heading and j == 0)


def chunk_pages(
    pages: Sequence[tuple[int | None, str]],
    chunk_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 20,
) -> list[TextChunk]:
    """Gabungkan paragraf menjadi chunk ~chunk_words kata dengan overlap.

    - Paragraf utuh dipertahankan bila muat; paragraf panjang dipecah per kalimat.
    - Judul bagian utama (BAB / "A. JUDUL") selalu memulai chunk baru; judul
      seksi lain (Pasal/Bagian) memulai chunk baru bila chunk sudah cukup terisi.
      Overlap tidak pernah melintasi batas seksi.
    - Setiap chunk diberi awalan "[Bagian: ...]" berisi judul bagian yang
      diwarisi (BAB › subjudul › Pasal) agar potongan tetap bermakna sendiri.
    - Nomor halaman awal/akhir dicatat untuk sitasi.
    """
    groups: list[tuple[list[_Unit], int]] = []  # (units, jumlah unit overlap di depan)
    ctx_before: dict[int, tuple] = {}  # id(unit) -> (l1, l2-list, l3) sebelum unit tsb
    l1: str | None = None
    l2: list[str] = []
    l3: str | None = None
    cur: list[_Unit] = []
    cur_n = 0
    carried = 0  # unit overlap yang dibawa dari chunk sebelumnya
    n_new = 0

    for u in _units(pages, chunk_words):
        if u.level == 2 and l1 is None:
            u.level = 0  # huruf kapital sebelum ada BAB/bagian utama (judul dokumen, nama pejabat)
        ctx_before[id(u)] = (l1, tuple(l2), l3)
        if u.level == 1:
            l1, l2, l3 = u.text, [], None
        elif u.level == 2:
            l2 = [x for x in l2 if x != u.text][-1:] + [u.text]
            l3 = None
        elif u.level == 3:
            l3 = u.text
        major = u.level == 1 and n_new > 0
        starts_section = major or (u.heading and cur_n >= 0.4 * chunk_words)
        if cur and (cur_n + u.n > chunk_words or starts_section):
            if n_new:
                groups.append((list(cur), carried))
            if starts_section or not n_new:
                cur, cur_n, carried = [], 0, 0
            else:
                tail: list[_Unit] = []
                tw = 0
                for x in reversed(cur):
                    if tw + x.n > overlap_words:
                        break
                    tail.append(x)
                    tw += x.n
                tail.reverse()
                cur, cur_n, carried = tail, tw, len(tail)
                if cur_n + u.n > chunk_words:
                    cur, cur_n, carried = [], 0, 0
            n_new = 0
        cur.append(u)
        cur_n += u.n
        n_new += 1
    if cur and n_new:
        groups.append((list(cur), carried))

    # Chunk terakhir yang terlalu kecil digabung ke chunk sebelumnya (tidak lintas bagian utama).
    if len(groups) >= 2:
        last_units, last_carried = groups[-1]
        new_part = last_units[last_carried:]
        if sum(u.n for u in new_part) < min_words and not any(u.level == 1 for u in new_part):
            groups.pop()
            prev_units, prev_carried = groups[-1]
            groups[-1] = (prev_units + new_part, prev_carried)

    chunks: list[TextChunk] = []
    for units, _ in groups:
        pages_ = [u.page for u in units if u.page is not None]
        # konteks = keadaan judul sebelum unit isi pertama, dikurangi judul yang tertulis di chunk
        j = next((i for i, u in enumerate(units) if not u.level), len(units) - 1)
        c1, c2, c3 = ctx_before.get(id(units[j]), (None, (), None))
        written = {u.text for u in units[:j]}
        parts = [x for x in [c1, *c2, c3] if x and x not in written]
        body = "\n".join(u.text for u in units)
        chunks.append(
            TextChunk(
                text=f"[Bagian: {' › '.join(parts)}]\n{body}" if parts else body,
                page_start=min(pages_) if pages_ else None,
                page_end=max(pages_) if pages_ else None,
                n_words=sum(u.n for u in units),
            )
        )
    return chunks
