"""Generator data dummy (sintetis) untuk testing, benchmarking, dan profiling.

Menghasilkan dokumen mirip peraturan akademik (BAB/Pasal/ayat) dengan entitas
FIKTIF (program studi & fakultas rekaan), sehingga tidak bisa tertukar dengan
informasi resmi. Setiap dokumen memuat satu "fakta kunci" unik yang menjadi
dasar pasangan tanya-jawab (ground truth) untuk mengukur Recall@k.

Mode:
  kb     Tulis langsung ke koleksi KB sintetis sampai target jumlah chunk
         tercapai (skala Tabel 3.1: ~1e4 / 1e5 / 1e6 chunk), lalu opsional
         bangun indeks: --index real (embedding model asli), fake (vektor acak,
         cepat — hanya untuk mengukur tahap pencarian), atau none.
  files  Tulis dokumen sebagai file .txt/.html/.pdf untuk menguji tahap
         parsing/ingestion (misal untuk `python -m tools.kb bench-build`).

Contoh:
  python -m tools.gen_dummy kb --collection dummy_s --chunks 10000 --index real
  python -m tools.gen_dummy kb --collection dummy_l --chunks 1000000 --index fake
  python -m tools.gen_dummy files --out data/dummy_files/pdf_500 --docs 500 --format pdf

Output tambahan: data/dummy/<nama>/qa.jsonl dan questions.txt (untuk uji beban).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from rag.config import chunk_params, load_config
from rag.text import chunk_pages

# ------------------------------------------------------------------ kosakata fiktif
FIELDS = [
    "Komputasi Kelautan", "Agroinformatika", "Bioinformatika Tropis", "Statistika Lingkungan",
    "Teknik Material Maju", "Sains Data Sosial", "Kesehatan Pesisir", "Rekayasa Energi Hijau",
    "Manajemen Kebencanaan", "Kecerdasan Buatan Terapan", "Ilmu Pangan Fungsional", "Geomatika Perkotaan",
    "Ekonomi Digital", "Farmasi Industri", "Teknologi Hutan Lestari", "Akuakultur Cerdas",
    "Kriptografi Terapan", "Robotika Pertanian", "Linguistik Komputasional", "Antropologi Digital",
    "Psikologi Siber", "Fisika Medis", "Kimia Hijau", "Mikrobiologi Pangan", "Arsitektur Tropis",
    "Oseanografi Terapan", "Teknik Biomedis", "Sistem Logistik", "Desain Interaksi", "Ekologi Urban",
    "Hidrologi Terapan", "Astronomi Komputasi", "Etnomusikologi Digital", "Gizi Olahraga",
    "Kebijakan Iklim", "Teknik Mekatronika", "Bahasa Isyarat", "Sejarah Maritim", "Vulkanologi",
    "Toksikologi Lingkungan",
]
VARIANTS = ["", " Terpadu", " Lanjut", " Internasional", " Berbasis Riset", " Kolaboratif", " Terapan Industri", " Interdisipliner"]
LEVELS = ["Sarjana", "Sarjana Terapan", "Magister", "Doktor"]
FACULTIES = [
    "Fakultas Sains Terapan Nusantara", "Fakultas Rekayasa Berkelanjutan", "Fakultas Ilmu Kehidupan Tropis",
    "Fakultas Sosial dan Humaniora Digital", "Fakultas Kesehatan Komunitas", "Fakultas Teknologi Cerdas",
    "Fakultas Bisnis dan Inovasi", "Fakultas Lingkungan dan Kebencanaan", "Sekolah Vokasi Terpadu",
    "Sekolah Interdisiplin Lanjut",
]
MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
          "November", "Desember"]

# topik: (judul, kalimat fakta kunci, pertanyaan, jawaban) — {x} diisi angka acak
TOPICS = {
    "cuti": ("Cuti Akademik",
             "Mahasiswa Program Studi {prodi} dapat mengajukan cuti akademik paling banyak {x} semester selama masa studi.",
             "Berapa semester paling banyak cuti akademik yang boleh diambil mahasiswa Program Studi {prodi} menurut peraturan tahun {tahun}?",
             "{x} semester", lambda r: r.choice([1, 2, 3, 4])),
    "krs": ("Pengisian Kartu Rencana Studi",
            "Batas akhir pengisian KRS bagi mahasiswa Program Studi {prodi} adalah {x} hari kerja setelah perkuliahan dimulai.",
            "Kapan batas akhir pengisian KRS untuk mahasiswa Program Studi {prodi} berdasarkan peraturan tahun {tahun}?",
            "{x} hari kerja", lambda r: r.randint(3, 21)),
    "beban_sks": ("Beban Studi Semester",
                  "Mahasiswa Program Studi {prodi} dengan IP semester di bawah 2,50 hanya boleh mengambil paling banyak {x} SKS pada semester berikutnya.",
                  "Berapa SKS maksimum untuk mahasiswa Program Studi {prodi} yang IP semesternya di bawah 2,50 menurut peraturan tahun {tahun}?",
                  "{x} SKS", lambda r: r.choice([12, 14, 15, 16, 18])),
    "ukt": ("Keringanan Uang Kuliah Tunggal",
            "Permohonan keringanan UKT bagi mahasiswa Program Studi {prodi} diajukan paling lambat {x} hari sebelum batas akhir pembayaran.",
            "Paling lambat berapa hari sebelum batas pembayaran permohonan keringanan UKT Program Studi {prodi} diajukan menurut peraturan tahun {tahun}?",
            "{x} hari", lambda r: r.randint(5, 45)),
    "yudisium": ("Penyelenggaraan Yudisium",
                 "Yudisium Program Studi {prodi} diselenggarakan setiap {x} bulan sekali.",
                 "Berapa bulan sekali yudisium Program Studi {prodi} diselenggarakan berdasarkan peraturan tahun {tahun}?",
                 "{x} bulan", lambda r: r.choice([1, 2, 3, 4, 6])),
    "beasiswa": ("Persyaratan Penerima Beasiswa",
                 "Penerima beasiswa pada Program Studi {prodi} wajib mempertahankan IPK minimal {x}.",
                 "Berapa IPK minimal yang wajib dipertahankan penerima beasiswa Program Studi {prodi} menurut peraturan tahun {tahun}?",
                 "{x}", lambda r: r.choice(["2,75", "3,00", "3,25", "3,50"])),
    "skripsi": ("Tugas Akhir dan Skripsi",
                "Bobot tugas akhir pada Program Studi {prodi} ditetapkan sebesar {x} SKS.",
                "Berapa SKS bobot tugas akhir di Program Studi {prodi} menurut peraturan tahun {tahun}?",
                "{x} SKS", lambda r: r.choice([4, 6, 8, 10])),
    "kkn": ("Kuliah Kerja Nyata",
            "Mahasiswa Program Studi {prodi} dapat mendaftar KKN setelah menempuh paling sedikit {x} SKS.",
            "Minimal berapa SKS yang harus ditempuh mahasiswa Program Studi {prodi} sebelum mendaftar KKN menurut peraturan tahun {tahun}?",
            "{x} SKS", lambda r: r.choice([90, 100, 110, 120])),
    "pindah": ("Perpindahan Program Studi",
               "Permohonan pindah ke Program Studi {prodi} hanya dapat diajukan oleh mahasiswa yang telah menempuh paling sedikit {x} semester.",
               "Setelah berapa semester mahasiswa boleh mengajukan pindah ke Program Studi {prodi} menurut peraturan tahun {tahun}?",
               "{x} semester", lambda r: r.choice([2, 3, 4])),
    "magang": ("Program Magang",
               "Durasi magang wajib bagi mahasiswa Program Studi {prodi} adalah {x} minggu.",
               "Berapa minggu durasi magang wajib untuk mahasiswa Program Studi {prodi} berdasarkan peraturan tahun {tahun}?",
               "{x} minggu", lambda r: r.choice([4, 6, 8, 12, 16])),
    "remedial": ("Ujian Perbaikan",
                 "Mahasiswa Program Studi {prodi} hanya dapat mengikuti ujian perbaikan untuk mata kuliah dengan nilai paling tinggi {x}.",
                 "Nilai paling tinggi berapa yang masih boleh diperbaiki lewat ujian perbaikan di Program Studi {prodi} menurut peraturan tahun {tahun}?",
                 "{x}", lambda r: r.choice(["C", "C+", "D", "D+"])),
    "masa_studi": ("Masa Studi dan Evaluasi Keberhasilan Studi",
                   "Masa studi maksimum bagi mahasiswa Program Studi {prodi} adalah {x} semester.",
                   "Berapa semester masa studi maksimum mahasiswa Program Studi {prodi} menurut peraturan tahun {tahun}?",
                   "{x} semester", lambda r: r.choice([8, 10, 12, 14])),
}

FILLER = [
    "Penyelenggaraan kegiatan akademik di lingkungan {fak} dilaksanakan berdasarkan prinsip akuntabilitas, transparansi, dan mutu.",
    "Ketua program studi bertanggung jawab atas pelaksanaan ketentuan ini dan melaporkannya kepada dekan setiap akhir semester.",
    "Mahasiswa wajib berkonsultasi dengan dosen pembimbing akademik sebelum melakukan perubahan rencana studi.",
    "Seluruh permohonan diajukan secara daring melalui sistem informasi akademik dengan melampirkan dokumen pendukung.",
    "Dokumen pendukung sebagaimana dimaksud meliputi surat permohonan, transkrip nilai sementara, dan bukti pembayaran.",
    "Keputusan atas permohonan disampaikan kepada mahasiswa paling lambat {n} hari kerja sejak berkas dinyatakan lengkap.",
    "Dalam hal terdapat keadaan kahar, dekan dapat menetapkan kebijakan khusus dengan memperhatikan pertimbangan senat fakultas.",
    "Evaluasi pelaksanaan peraturan ini dilakukan oleh unit penjaminan mutu paling sedikit {n} kali dalam satu tahun akademik.",
    "Mahasiswa yang tidak memenuhi ketentuan sebagaimana dimaksud pada ayat sebelumnya diberikan peringatan tertulis.",
    "Sanksi administratif dapat berupa teguran, penundaan layanan akademik, atau pembatalan kegiatan yang telah diikuti.",
    "Ketentuan teknis mengenai tata cara pelaksanaan diatur lebih lanjut dalam pedoman yang ditetapkan oleh dekan.",
    "Pembimbing akademik memantau kemajuan studi mahasiswa bimbingannya dan memberikan rekomendasi secara berkala.",
    "Layanan administrasi akademik diselenggarakan pada hari kerja pukul 08.00 sampai dengan 15.00 waktu setempat.",
    "Data akademik mahasiswa dikelola secara terintegrasi dan dijaga kerahasiaannya sesuai ketentuan yang berlaku.",
    "Program studi wajib menyampaikan informasi mengenai ketentuan ini kepada mahasiswa baru pada masa orientasi.",
    "Kegiatan pembelajaran dapat dilaksanakan secara luring, daring, atau bauran sesuai karakteristik mata kuliah.",
    "Penilaian hasil belajar dilakukan secara objektif, transparan, dan dapat dipertanggungjawabkan.",
    "Mahasiswa berhak mengajukan keberatan atas hasil penilaian paling lambat {n} hari setelah nilai diumumkan.",
]
DEFINITIONS = [
    "Mahasiswa adalah peserta didik yang terdaftar secara sah pada program studi di lingkungan {fak}.",
    "Semester adalah satuan waktu kegiatan pembelajaran efektif selama paling sedikit 16 minggu.",
    "Satuan Kredit Semester yang selanjutnya disingkat SKS adalah takaran waktu kegiatan belajar mahasiswa.",
    "Indeks Prestasi Semester adalah ukuran keberhasilan belajar mahasiswa pada satu semester.",
    "Dosen Pembimbing Akademik adalah dosen yang ditugaskan membimbing mahasiswa dalam perencanaan studi.",
    "Sistem informasi akademik adalah sistem elektronik yang digunakan untuk layanan administrasi akademik.",
]


def prodi_pool() -> list[str]:
    return [f"{lvl} {f}{v}" for lvl in LEVELS for f in FIELDS for v in VARIANTS]


class DocGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.r = random.Random(seed)
        self.prodis = prodi_pool()
        self.used: set[tuple[str, str, int]] = set()
        self.counter = 0

    def _key(self) -> tuple[str, str, int]:
        for _ in range(1000):
            k = (self.r.choice(self.prodis), self.r.choice(list(TOPICS)), self.r.randint(1995, 2035))
            if k not in self.used:
                self.used.add(k)
                return k
        raise RuntimeError("Kombinasi unik habis; kurangi jumlah dokumen")

    def generate(self) -> dict:
        r = self.r
        prodi, topic, tahun = self._key()
        title_topic, fact_t, q_t, a_t, xfn = TOPICS[topic]
        fak = r.choice(FACULTIES)
        nomor = r.randint(1, 99)
        x = xfn(r)
        self.counter += 1
        title = f"[SINTETIS] Peraturan Dekan {fak} Nomor {nomor} Tahun {tahun} tentang {title_topic} Program Studi {prodi}"
        fill = lambda s: s.format(fak=fak, prodi=prodi, n=r.randint(2, 30))  # noqa: E731
        paras: list[str] = [
            f"PERATURAN DEKAN {fak.upper()} NOMOR {nomor} TAHUN {tahun} TENTANG {title_topic.upper()}",
            f"Menimbang bahwa untuk menjamin mutu penyelenggaraan {title_topic.lower()} pada Program Studi {prodi} "
            f"perlu menetapkan peraturan dekan.",
            "BAB I KETENTUAN UMUM", "Pasal 1",
        ]
        paras += [fill(d) for d in r.sample(DEFINITIONS, k=r.randint(3, len(DEFINITIONS)))]
        n_pasal = r.randint(4, 12)
        fact_pasal = r.randint(2, n_pasal)
        fact_sentence = fact_t.format(prodi=prodi, x=x)
        for p in range(2, n_pasal + 1):
            if p % 4 == 2:
                paras.append(f"BAB {['II', 'III', 'IV', 'V', 'VI', 'VII'][min(5, p // 4)]} {title_topic.upper()}")
            paras.append(f"Pasal {p}")
            ayat = [fill(s) for s in r.sample(FILLER, k=r.randint(2, 5))]
            if p == fact_pasal:
                ayat.insert(r.randint(0, len(ayat)), fact_sentence)
            paras += [f"({i}) {s}" for i, s in enumerate(ayat, 1)]
        paras.append(f"Ditetapkan pada tanggal {r.randint(1, 28)} {r.choice(MONTHS)} {tahun}.")
        return {
            "doc_key": f"synthetic:{self.counter:08d}",
            "title": title,
            "topic": topic,
            "text": "\n\n".join(paras),
            "qa": {
                "question": q_t.format(prodi=prodi, tahun=tahun),
                "answer": a_t.format(x=x),
                "fact": fact_sentence,
                "title": title,
            },
        }


# ------------------------------------------------------------------ mode kb
def run_kb(a) -> None:
    from rag.indexer import build_bm25, save_index
    from rag.service import RAGService
    from rag.store import KnowledgeBase

    svc = RAGService()
    if not a.collection.startswith(("dummy", "synthetic", "bench")):
        sys.exit("Nama koleksi dummy harus diawali 'dummy', 'synthetic', atau 'bench' agar tidak tertukar dengan KB resmi.")
    if a.overwrite:
        KnowledgeBase.drop(svc.kb_root, a.collection)
    svc.create_collection(a.collection, f"Data sintetis ~{a.chunks} chunk (seed {a.seed})", synthetic=True)
    kb = svc.kb(a.collection)
    if not kb.synthetic:
        sys.exit("Koleksi sudah ada dan bukan sintetis — batal.")
    ccfg = chunk_params(load_config())
    gen = DocGenerator(a.seed)
    qa_all: list[dict] = []
    t0 = time.perf_counter()
    state = {"chunks": kb.stats()["n_chunks"]}

    def docs():
        while state["chunks"] < a.chunks:
            d = gen.generate()
            chunks = chunk_pages([(1, d["text"])], **ccfg)
            state["chunks"] += len(chunks)
            qa_all.append({**d["qa"], "doc_key": d["doc_key"], "topic": d["topic"]})
            if gen.counter % 5000 == 0:
                print(f"  {gen.counter} dokumen, {state['chunks']} chunk ({time.perf_counter() - t0:.0f}s)")
            yield {"doc_key": d["doc_key"], "title": d["title"], "chunks": chunks, "category": d["topic"],
                   "n_pages": 1, "n_chars": len(d["text"]), "source_type": "synthetic", "meta": {"synthetic": True}}

    n = kb.bulk_add_documents(docs())
    print(f"Generate+chunk+simpan: {gen.counter} dokumen, {n} chunk dalam {time.perf_counter() - t0:.1f}s")
    write_qa(Path(load_config()["paths"]["data_dir"]) / "dummy" / a.collection, qa_all, a.qa, a.seed)

    if a.index == "real":
        meta = svc.rebuild(a.collection, full=True, mode=a.mode, log_fn=print)
        print(f"Indeks (embedding asli): {meta['timings_ms']}")
    elif a.index == "fake":
        t = time.perf_counter()
        ids = kb.all_chunk_ids()
        dim = a.dim
        rng = np.random.default_rng(a.seed)
        emb = np.empty((len(ids), dim), dtype=np.float32)
        for i in range(0, len(ids), 100_000):
            blk = rng.standard_normal((min(100_000, len(ids) - i), dim), dtype=np.float32)
            blk /= np.linalg.norm(blk, axis=1, keepdims=True)
            emb[i : i + len(blk)] = blk
        t_emb = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        bm25 = build_bm25(kb, ids)
        t_bm25 = (time.perf_counter() - t) * 1000
        emb_name = load_config()["models"]["embedder"]["name"]
        save_index(kb, ids, emb, bm25, {
            "collection": kb.name, "embed_model": f"RANDOM({emb_name})", "embed_device": "none",
            "embed_dtype": "float32", "mode": "fake", "fake_embeddings": True, "synthetic": True,
            "timings_ms": {"embed_fake": round(t_emb, 1), "bm25": round(t_bm25, 1)},
        })
        print(f"Indeks palsu: {len(ids)} vektor acak dim={dim} ({t_emb:.0f} ms), BM25 {t_bm25:.0f} ms")
        print("  Catatan: vektor acak -> hasil pencarian vektor tidak bermakna; hanya untuk mengukur latensi.")


def write_qa(out_dir: Path, qa_all: list[dict], n: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    sample = rnd.sample(qa_all, k=min(n, len(qa_all))) if qa_all else []
    with open(out_dir / "qa.jsonl", "w", encoding="utf-8") as f:
        for q in sample:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (out_dir / "questions.txt").write_text("\n".join(q["question"] for q in sample) + "\n", encoding="utf-8")
    print(f"QA ground truth: {out_dir / 'qa.jsonl'} ({len(sample)} pasang)")


# ------------------------------------------------------------------ mode files
def run_files(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = DocGenerator(a.seed)
    qa_all = []
    t0 = time.perf_counter()
    for i in range(a.docs):
        d = gen.generate()
        qa_all.append({**d["qa"], "doc_key": d["doc_key"], "topic": d["topic"]})
        name = f"sintetis_{i:06d}"
        if a.format == "txt":
            (out / f"{name}.txt").write_text(d["title"] + "\n\n" + d["text"], encoding="utf-8")
        elif a.format == "html":
            body = "".join(f"<p>{p}</p>" for p in d["text"].split("\n\n"))
            html = (f"<!doctype html><html><head><title>{d['title']}</title></head><body><nav>Menu Beranda Kontak</nav>"
                    f"<main><article><h1>{d['title']}</h1>{body}</article></main><footer>Hak cipta fiktif</footer></body></html>")
            (out / f"{name}.html").write_text(html, encoding="utf-8")
        else:
            write_pdf(out / f"{name}.pdf", d["title"], d["text"])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{a.docs} file")
    print(f"{a.docs} file {a.format} ditulis ke {out} dalam {time.perf_counter() - t0:.1f}s")
    write_qa(out.parent / f"{out.name}_qa", qa_all, a.qa, a.seed)


def write_pdf(path: Path, title: str, text: str) -> None:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # type: ignore
    doc = pymupdf.open()
    paras = [title] + text.split("\n\n")
    page = None
    y = 0.0
    for p in paras:
        need = 14 * (1 + len(p) // 85)
        if page is None or y + need > 780:
            page = doc.new_page(width=595, height=842)
            page.insert_text((60, 30), "SALINAN — DOKUMEN SINTETIS", fontsize=8)
            page.insert_text((280, 820), f"{doc.page_count}", fontsize=8)
            y = 60.0
        rect = pymupdf.Rect(60, y, 535, y + need + 6)
        page.insert_textbox(rect, p, fontsize=10, fontname="helv")
        y += need + 8
    doc.set_metadata({"title": title, "producer": "gen_dummy (data sintetis)"})
    doc.save(path)
    doc.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kb", help="tulis langsung ke koleksi KB sintetis")
    p.add_argument("--collection", required=True)
    p.add_argument("--chunks", type=int, default=10_000, help="target jumlah chunk")
    p.add_argument("--index", choices=["real", "fake", "none"], default="real")
    p.add_argument("--dim", type=int, default=384, help="dimensi vektor acak untuk --index fake")
    p.add_argument("--mode", default="baseline")
    p.add_argument("--qa", type=int, default=500, help="jumlah pasangan QA yang disimpan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="hapus koleksi lama dengan nama sama")
    p = sub.add_parser("files", help="tulis dokumen sebagai file")
    p.add_argument("--out", required=True)
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--format", choices=["txt", "html", "pdf"], default="txt")
    p.add_argument("--qa", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run_kb(a) if a.cmd == "kb" else run_files(a)


if __name__ == "__main__":
    main()
