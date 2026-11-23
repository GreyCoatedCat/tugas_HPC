"""Pembersihan teks hasil ekstraksi PDF/HTML.

Langkah utama:
1. Normalisasi Unicode (NFKC), buang karakter tak terlihat & kontrol.
2. PDF: buang header/footer yang berulang di banyak halaman dan nomor halaman.
3. PDF: gabungkan kembali baris yang terpotong (reflow) menjadi paragraf,
   termasuk memperbaiki pemenggalan kata dengan tanda hubung.
4. HTML: buang baris boilerplate (tombol share, "baca juga", dsb.) dan duplikat.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t  -   　]+")
_BULLET_RE = re.compile(r"[▪●◦•‣⁃∙]")

_PAGE_NO_RE = re.compile(
    r"^(?:-\s*)?(?:(?:hal(?:aman)?|page)\.?\s*)?\d{1,4}(?:\s*(?:dari|/|of)\s*\d{1,4})?(?:\s*-)?$",
    re.IGNORECASE,
)
_PARA_START_RE = re.compile(
    r"^(?:\(?\d{1,3}[.)]\s|\(?[A-Za-z][.)]\s|[•\-–]\s|"
    r"(?:BAB|Pasal|Bagian|Paragraf|Menimbang|Mengingat|Memperhatikan|Menetapkan|MEMUTUSKAN|"
    r"KESATU|KEDUA|KETIGA|KEEMPAT|KELIMA|LAMPIRAN)\b)"
)
_MONTHS = "Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember"
# baris tabel jadwal/kalender biasanya diawali tanggal: "1 Juli 2026 - 31 Juli 2026 Heregistrasi ..."
_DATE_START_RE = re.compile(rf"^\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{4}})?\b", re.IGNORECASE)
_HEADING_ONLY_RE = re.compile(r"^(?:BAB|Pasal|Bagian|Paragraf)\s+[\w.]+$", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"^(?:share|bagikan|facebook|twitter|x|whatsapp|linkedin|telegram|email|e-mail|print|cetak|"
    r"copy link|salin tautan|baca juga\b.*|tags?\s*:.*|kategori\s*:.*|posted in\b.*|"
    r"previous( post)?|next( post)?|sebelumnya|selanjutnya|skip to content|menu|search|cari)$",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_RE.sub(" ", text)
    return _BULLET_RE.sub("•", text)


def _signature(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def strip_repeated_edges(pages_lines: list[list[str]], edge: int = 3, min_ratio: float = 0.5) -> list[list[str]]:
    """Buang baris header/footer yang muncul berulang di tepi banyak halaman."""
    n = len(pages_lines)
    if n < 3:
        return pages_lines
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        nz = [ln for ln in lines if ln.strip()]
        counts.update({_signature(ln) for ln in nz[:edge] + nz[-edge:]})
    threshold = max(3, int(min_ratio * n + 0.999))
    repeated = {s for s, c in counts.items() if c >= threshold and 0 < len(s) <= 150}
    if not repeated:
        return pages_lines
    out = []
    for lines in pages_lines:
        nz_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(nz_idx[:edge] + nz_idx[-edge:])
        out.append([ln for i, ln in enumerate(lines) if not (i in edge_idx and _signature(ln) in repeated)])
    return out


def _join_hyphenated(buf: str, line: str) -> str:
    prev_word = re.findall(r"(\w+)-$", buf)
    next_word = re.match(r"(\w+)", line)
    if prev_word and next_word and prev_word[-1].lower() == next_word.group(1).lower():
        return buf + line  # reduplikasi: "anak-" + "anak" -> "anak-anak"
    return buf[:-1] + line


def reflow(lines: list[str]) -> str:
    """Gabungkan baris-baris hasil ekstraksi PDF menjadi paragraf utuh."""
    lines = [_WS_RE.sub(" ", ln).strip() for ln in lines]
    lengths = sorted(len(ln) for ln in lines if ln)
    full_len = lengths[int(0.8 * (len(lengths) - 1))] if lengths else 0
    paras: list[str] = []
    buf = ""
    prev_len = 0
    for line in lines:
        if not line:
            if buf:
                paras.append(buf)
                buf = ""
            prev_len = 0
            continue
        if _PAGE_NO_RE.match(line):
            continue
        if not buf:
            buf, prev_len = line, len(line)
            continue
        prev_short = (full_len >= 40 and prev_len < 0.6 * full_len) or bool(_HEADING_ONLY_RE.match(buf))
        ends_sentence = buf[-1] in ".:;?!" and (line[0].isupper() or line[0].isdigit() or line[0] in "(•")
        if _PARA_START_RE.match(line) or _DATE_START_RE.match(line) or prev_short or ends_sentence:
            paras.append(buf)
            buf = line
        elif buf.endswith("-") and line[:1].islower():
            buf = _join_hyphenated(buf, line)
        else:
            buf = f"{buf} {line}"
        prev_len = len(line)
    if buf:
        paras.append(buf)
    return "\n\n".join(paras)


def clean_pdf_pages(raw_pages: list[str]) -> list[str]:
    pages_lines = [normalize_unicode(p).split("\n") for p in raw_pages]
    pages_lines = strip_repeated_edges(pages_lines)
    return [reflow(lines) for lines in pages_lines]


def clean_plain_text(text: str) -> str:
    """Untuk teks HTML (sudah per paragraf): buang boilerplate & duplikat."""
    out: list[str] = []
    seen_recent: list[str] = []
    for line in normalize_unicode(text).split("\n"):
        line = _WS_RE.sub(" ", line).strip()
        if not line:
            if out and out[-1]:
                out.append("")
            continue
        if _BOILERPLATE_RE.match(line) or line in seen_recent:
            continue
        out.append(line)
        seen_recent = (seen_recent + [line])[-5:]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def clean_wrapped_text(text: str) -> str:
    """Untuk file .txt yang barisnya di-wrap manual."""
    return reflow(normalize_unicode(text).split("\n"))


def text_stats(text: str) -> dict[str, float]:
    chars = len(text)
    alpha = sum(ch.isalpha() for ch in text)
    return {
        "chars": chars,
        "words": len(text.split()),
        "alpha_ratio": round(alpha / chars, 3) if chars else 0.0,
    }
