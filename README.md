# Asisten Informasi Akademik UGM — Pipeline RAG + Akselerasi OpenMP/CUDA

Implementasi proposal *"Akselerasi Pipeline Retrieval-Augmented Generation pada
Asisten Informasi Akademik UGM menggunakan OpenMP dan CUDA"*: chatbot RAG lokal
yang **hanya menjawab dari knowledge base dokumen resmi**, dengan dua mode
yang bisa dialihkan langsung di aplikasi:

- **baseline**: pipeline sederhana, sekuensial, dan menjadi titik acuan;
- **optimized**: hasil optimasi berbasis profiling, yaitu kernel CUDA top-k
  dan BM25 OpenMP (C++, `cpp/`), FP16, dynamic batching + overlap CPU–GPU,
  parsing paralel, dan LLM dengan continuous batching.

Proyek ini juga dilengkapi scraper, generator data dummy, alat uji beban,
ablation, dan laporan sebelum–sesudah (bagian *Optimasi* di bawah).

```
            Fase offline                                  Fase online (per pertanyaan)
 Dokumen resmi ─► parse+clean ─► chunk ─► embedding ─►  embed kueri ─► vektor top-k ─┐
 (PDF/HTML)      [proses paralel,        [batch FP16]    (e5-small)    [CUDA]        │
                  pecah per halaman]      + BM25                       BM25 top-k  ──┤ RRF ─► rerank ─► [ambang
                                          indeks                       [OpenMP]      ┘     (cross-encoder,   relevansi]
                                                                                            FP16, batch)       │
                    jawaban + sitasi [n] ◄─ LLM qwen3.5:9b (llama.cpp, continuous batching) ◄────────────────┘
```

## Struktur

| Folder | Isi |
|---|---|
| `rag/` | inti pipeline: cleaning, chunking, BM25, pencarian vektor, wrapper model (embedder, reranker, LLM), penyimpanan KB, service + timing per tahap |
| `scraper/` | scraping dokumen UGM (patuh robots.txt, rate-limit) + cleaning; daftar sumber di `seeds.yaml` |
| `server/` | API FastAPI (dipakai UI, uji beban, dan tools) |
| `ui/` | demo Gradio: chat + sumber + latensi per tahap, manajemen KB, monitor & uji beban |
| `tools/` | CLI knowledge base (`kb.py`), generator data dummy (`gen_dummy.py`), Recall@k (`eval_recall.py`), server LLM teroptimasi (`llm_server.py`), probe LLM (`llm_probe.py`), ablation (`ablation.py`), laporan sebelum–sesudah (`compare.py`) |
| `loadtest/` | skenario Locust, orkestrator sapuan konkurensi (`run.py`), benchmark satu mode (`run_full.sh`), evaluasi sebelum–sesudah (`run_compare.sh`) |
| `cpp/` | modul C++ `rag_hpc`: kernel CUDA similarity + top-k, BM25 OpenMP (pybind11) — lihat [cpp/README.md](cpp/README.md) |
| `tests/` | uji otomatis (pytest): kesetaraan CUDA vs NumPy, OpenMP vs SciPy, ingest paralel, antrean FIFO, retrieval optimized vs baseline |
| `docs/` | [isian bagian merah proposal](docs/isian_proposal.md) dari hasil pengukuran |
| `demo.sh` | menyalakan server LLM + API + UI dengan satu perintah |
| `config.yaml` | semua parameter (model, retrieval, ambang relevansi, mode baseline/optimized) |
| `data/` | KB, hasil scraping, data dummy (dibuat otomatis, tidak di-commit) |
| `results/` | log request, hasil uji beban, build indeks, recall |

## Instalasi

Prasyarat: Python 3.12, GPU NVIDIA (driver ≥ 580 untuk wheel torch cu13x),
[Ollama](https://ollama.com).

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
ollama pull qwen3.5:9b
```

Bangun modul C++ (CUDA + OpenMP). Butuh nvcc, CMake, Ninja, dan g++-12; detailnya
ada di [cpp/README.md](cpp/README.md):

```bash
bash cpp/build.sh
```

Model embedding (`intfloat/multilingual-e5-small`) dan reranker
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) diunduh otomatis dari Hugging
Face saat pertama dipakai (±1 GB).

## Menjalankan

Cara cepat, satu perintah. Skrip ini menyalakan server LLM, API, dan UI berurutan,
menunggu masing-masing siap, dan Ctrl+C mematikan semuanya:

```bash
bash demo.sh
```

Atau manual, dengan satu terminal per langkah:

1. Server LLM. Perintah ini menjalankan `llama-server` bawaan Ollama dengan berkas
   model yang sama, 4 slot paralel, dan semua layer di GPU. Alasannya ada di
   bagian *Optimasi*. Butuh model hasil `ollama pull`; daemon Ollama sendiri
   tidak perlu berjalan.

```bash
.venv/bin/python -m tools.llm_server
```

2. Isi knowledge base resmi (scrape + clean + ingest + bangun indeks). Cukup
   sekali, atau setiap kali `seeds.yaml` berubah:

```bash
.venv/bin/python -m scraper --ingest ugm
```

3. API server (memuat model + warmup, ±20 detik):

```bash
.venv/bin/python -m server
```

4. UI demo → buka http://127.0.0.1:7860

```bash
.venv/bin/python -m ui
```

## Knowledge base & aturan "hanya dari dokumen resmi"

Jawaban dijaga berlapis agar tidak keluar dari knowledge base:

1. **Ambang relevansi sebelum LLM.** Setiap kandidat dinilai cross-encoder
   reranker. Chunk dengan skor < `retrieval.min_rerank_score` dibuang. Jika tidak
   ada yang lolos, **LLM tidak dipanggil sama sekali** dan chatbot membalas
   pesan penolakan baku (`grounding.refuse_message`).
2. **Prompt ketat.** LLM hanya boleh memakai teks di KONTEKS, wajib memberi
   sitasi `[n]` untuk setiap fakta, dan wajib membalas pesan penolakan bila
   konteks tidak memuat jawaban. Teks dokumen diperlakukan sebagai data, bukan instruksi.
3. **Pemeriksaan sesudah generasi.** Sitasi di jawaban dicocokkan dengan sumber.
   Jawaban tanpa sitasi diberi peringatan di UI. Jawaban penolakan
   menyembunyikan panel sumber.
4. **Koleksi terpisah.** Data dummy hanya boleh masuk koleksi bertanda
   `synthetic` (nama harus diawali `dummy`/`synthetic`/`bench`) dan ditandai
   jelas di UI, sehingga tidak pernah tercampur dengan dokumen resmi.
5. **Konteks bagian.** Setiap chunk membawa judul bagian yang diwarisi, misalnya
   `[Bagian: BAB II › PENERIMAAN MAHASISWA › Pasal 3]` atau
   `[Bagian: B. KALENDER AKADEMIK PROGRAM PASCASARJANA › SEMESTER GASAL]`,
   agar LLM tidak mencampur ketentuan antarprogram atau antarbab.

**Menambah dokumen:**

- **UI** → tab *Knowledge Base*: unggah PDF/HTML/TXT/MD (opsional judul,
  kategori, URL sumber resmi untuk sitasi), atau *Tambah dari URL resmi*.
  Indeks diperbarui inkremental (hanya chunk baru yang di-embed).
- **CLI:**

```bash
.venv/bin/python -m tools.kb ingest ugm path/ke/dokumen.pdf folder/lain --category peraturan
```

- **Scraper:** tambah entri di `scraper/seeds.yaml`, lalu jalankan
  `python -m scraper --ingest ugm`.
- **API:** `POST /api/kb/{koleksi}/documents` (multipart) atau
  `POST /api/kb/{koleksi}/url`.

Perintah CLI lain: `python -m tools.kb list | docs ugm | delete ugm <id> | rebuild ugm --full | ask ugm "pertanyaan"`.

## Scraping & cleaning

```bash
.venv/bin/python -m scraper                 # ambil semua seed → data/scrape/{raw,clean}, report.md
```

```bash
.venv/bin/python -m scraper --reclean       # ulangi cleaning dari file raw (setelah mengubah aturan cleaning)
```

Cleaning (`rag/cleaning.py`):
- normalisasi Unicode;
- buang header/footer yang berulang di banyak halaman serta nomor halaman;
- gabungkan baris PDF yang terpotong menjadi paragraf, termasuk memperbaiki
  pemenggalan kata tanpa merusak reduplikasi seperti "anak-anak";
- pisahkan butir ayat/huruf/Pasal/BAB dan baris tabel bertanggal;
- untuk HTML: ekstraksi konten utama dengan trafilatura, lalu buang boilerplate.

`data/scrape/report.md` mencatat jumlah karakter raw vs. bersih per dokumen.
Seed awal berisi 14 dokumen resmi: PR 23/2024 tentang Pendidikan + perubahannya,
Kalender Akademik 2026/2027, PMB, heregistrasi & UKT, beasiswa, kurikulum FMIPA,
dan berita resmi.

## Pipeline baseline & mode

Mode `baseline` (`config.yaml → modes.baseline`) mengikuti Bab 3.4 proposal:

| Tahap | Implementasi baseline |
|---|---|
| Parsing + chunking | sekuensial, 1 worker |
| Embedding dokumen | PyTorch FP32, **batch 1** |
| Pencarian vektor | brute-force NumPy, **1 thread BLAS** |
| BM25 | matriks sparse SciPy, akumulasi posting list sekuensial |
| Request | `pipeline_max_concurrency: 1`, `llm_max_concurrency: 1` (tanpa batching) |

Setiap request mencatat latensi per tahap: `queue_retrieval`, `embed_query`,
`search_vector`, `search_bm25`, `fusion`, `fetch_chunks`, `rerank`,
`queue_llm`, `llm_prefill` (s.d. token pertama), dan `llm_decode`, ditambah
TTFT dan e2e. Data ini dikirim ke UI, ke `/api/metrics`, dan ditulis ke
`results/requests/requests-YYYYMMDD.jsonl`.

Backend LLM bisa ditukar lewat `models.llm.backend`:
- `openai` (default): llama.cpp server dari `python -m tools.llm_server`, atau vLLM;
- `ollama`: konfigurasi baseline asli, ada sebagai komentar di config.yaml;
- `transformers`: HF langsung.

## Optimasi (mode `optimized`)

Setiap optimasi menjawab bottleneck yang terukur pada profiling baseline
(bagian *Hasil benchmark baseline* di bawah). Kualitas dijaga dan diverifikasi:
konteks yang dipilih, Recall@k, dan hasil ingest identik dengan baseline.

| Bottleneck (bukti profiling) | Teknik | Implementasi |
|---|---|---|
| Pencarian vektor 69 ms pada 10⁶ chunk (CPU 1 thread, O(N·d)) | **Kernel CUDA kustom**: similarity multi-kueri (warp per baris, kueri di shared memory, matriks FP16) + radix-select top-k di GPU | `cpp/src/topk_cuda.cu` |
| BM25 36 ms pada 10⁶ chunk (akumulasi sekuensial + `bincount` atas semua dokumen) | **OpenMP**: segmen dokumen per thread (`schedule(dynamic)`), tanpa atomik, top-k lokal lalu digabung | `cpp/src/bm25_omp.cpp` |
| Rerank ±120 ms = 97% retrieval; compute-bound | **FP16 (tensor core)** lewat autocast, tanpa salinan bobot tambahan di VRAM | `rag/models/reranker.py` |
| Satu slot retrieval melayani permintaan satu per satu (jenuh ±6,5 QPS) | **Dynamic batching**: permintaan yang menumpuk diproses dalam satu batch (embedding, kernel vektor, rerank) + **overlap** BM25 (CPU) ∥ vektor (GPU) | `rag/batching.py`, `RAGService.retrieve_batch` |
| Embedding = 66% build; batch 1 FP32 | **Batch 64 FP16**, diurutkan menurut panjang (padding minimal), tokenisasi batch berikutnya (CPU, pinned memory) tumpang-tindih dengan forward pass (GPU) | `rag/models/embedder.py` |
| Parsing = 32% build; satu buku 563 halaman = 7 detik | **Parsing paralel** di pool proses dengan penjadwalan LPT; PDF tebal **dipecah per 32 halaman** agar beban merata | `rag/ingest.py` |
| LLM decode > 95% e2e; satu permintaan per waktu; 9 dari 34 layer di CPU | **Continuous batching** 4 slot + **semua layer di GPU** (tanpa vision projector, flash attention, KV cache 8-bit) | `tools/llm_server.py` |

Catatan untuk LLM:
- Ollama menolak permintaan paralel untuk arsitektur `qwen35` (pesan di log:
  `model architecture does not currently support parallel requests`).
- Ollama juga menyisihkan ±1,3 GB VRAM untuk vision projector yang tidak
  dipakai, sehingga hanya 25 dari 34 layer masuk GPU.
- Karena itu dipakai `llama-server`, runner llama.cpp yang dipakai Ollama di
  balik layar, langsung dengan **berkas GGUF yang sama** (bobot identik).
- VRAM 8 GB hanya cukup untuk satu salinan model. Akibatnya kedua mode di demo
  berbagi server LLM ini dan berbeda dalam penjadwalan: baseline mengirim 1
  permintaan per waktu, optimized hingga 4 sekaligus.
- Angka "sebelum" tetap memakai baseline asli (Ollama bawaan, 20 Nov), dan ada
  run ablation "pipeline baseline + runtime baru" untuk memisahkan efek runtime
  dari efek batching.

Parameter tiap mode ada di `config.yaml → modes`. Status modul C++ tampil di
header UI dan `GET /api/health → hpc`.

### Demo langsung

1. Jalankan server LLM, API, dan UI seperti di bagian *Menjalankan*.
2. **Tab Chat.** Pilih mode **baseline** atau **optimized** dengan tombol
   radio, lalu ajukan pertanyaan. Panel *Latensi per tahap* menampilkan waktu
   setiap tahap. Di mode optimized ditampilkan juga ukuran batch, waktu kernel
   CUDA (similarity + top-k), BM25 OpenMP, dan penghematan overlap.
3. **Tombol "Bandingkan mode".** Pertanyaan yang sama dijalankan 5× di kedua
   mode (retrieval saja). Panel kanan menampilkan total percepatan, tabel
   percepatan per tahap, dan cek bahwa sumber konteksnya identik. Koleksi
   `dummy_m`/`dummy_l` memperlihatkan efek CUDA/OpenMP pada korpus besar, tetapi
   indeks GPU-nya baru muat bila server LLM tidak sedang memakai VRAM (jalankan
   `python -m server` dan `python -m ui` saja). Saat demo lengkap berjalan,
   bandingkan di koleksi `ugm`.
4. **Tab Monitor & Uji Beban.** Jalankan pengguna virtual (mis. 20 pengguna,
   dengan LLM) sambil tetap chat. Tabel monitor memisahkan metrik per mode
   (throughput, p50/p95, TTFT). Jalankan sekali di baseline lalu sekali di
   optimized: antrean LLM di baseline membengkak, sedangkan di optimized
   empat jawaban dikerjakan bersamaan.
5. **Tab Hasil Speedup.** Ringkasan pengukuran sebelum–sesudah yang sudah
   dijalankan: kartu angka utama, grafik `results/figures/`, tabel dari
   `results/comparison.md`, ablation + hukum Amdahl, micro-benchmark kernel,
   dan Recall@k baseline vs optimized. Halaman ini membaca berkas di
   `results/`, jadi muat ulang UI setelah mengulang pengukuran.

### Evaluasi sebelum–sesudah

```bash
bash loadtest/run_compare.sh     # ±2,5 jam: ablation + run_full.sh baseline & optimized + laporan
```

```bash
.venv/bin/python -m tools.ablation --collections ugm dummy_l    # kontribusi per optimasi
```

```bash
.venv/bin/python -m tools.compare                                # tulis ulang results/comparison.md
```

### Hasil sebelum–sesudah (23 Nov 2026)

Diukur dengan `bash loadtest/run_compare.sh` pada RTX 5050 Laptop 8 GB, i5-13450HX, RAM 15 GB:
- baseline dan optimized diukur pada sesi yang sama, rata-rata 3 pengulangan per titik, tanpa satu pun request gagal;
- laporan lengkap ada di [results/comparison.md](results/comparison.md);
- ablation ada di `results/ablation/ablation-20261123-140352.md`.

| Metrik | Baseline | Optimized | Percepatan |
|---|---:|---:|---:|
| Retrieval 10⁶ chunk, throughput jenuh (100 pengguna) | 3,9 QPS | 26,6 QPS | **6,8×** |
| Retrieval 10⁶ chunk, p95 (50 pengguna) | 11,0 s | 0,70 s | **15,8×** |
| Retrieval 10⁶ chunk, latensi 1 pengguna (p50) | 257 ms | 60 ms | **4,3×** |
| Retrieval KB resmi, throughput jenuh (50 pengguna) | 6,3 QPS | 13,9 QPS | **2,2×** |
| Retrieval KB resmi, latensi 1 pengguna (p50) | 140 ms | 79 ms | **1,8×** |
| End-to-end + LLM, throughput token (50 pengguna)¹ | 13,6 tok/s | 32,5 tok/s | **2,4×** |
| End-to-end + LLM, jawaban selesai (10–100 pengguna) | ±0,04/s | ±0,08/s | **2,0×** |
| Build indeks, 14 dokumen resmi | 30,6 s | 9,6 s | **3,2×** |
| Build indeks, 1.000 PDF | 64,0 s | 11,7 s | **5,5×** |

¹ Baseline asli (Ollama bawaan, 20 Nov). Throughput token dipakai karena parameter
sampling bawaan Ollama dan llama.cpp menghasilkan panjang jawaban yang berbeda
(median 297 vs 390–434 token).

**Per tahap, 10⁶ chunk, 1 pengguna (lewat HTTP):**

| Tahap | Baseline | Optimized | Percepatan |
|---|---:|---:|---:|
| Pencarian vektor (CUDA) | 109 ms | 5,3 ms | 20,5× |
| BM25 (OpenMP, 8 thread) | 52 ms | 5,9 ms | 8,9× |
| Rerank (FP16) | 79 ms | 35 ms | 2,2× |

**Ablation** (kumulatif, di dalam proses, latensi diukur bergiliran antar-konfigurasi):

| Konfigurasi | 10⁶ chunk: latensi | 10⁶ chunk: QPS jenuh | KB resmi: latensi | KB resmi: QPS jenuh |
|---|---:|---:|---:|---:|
| baseline | 188 ms | 5,3 | 148 ms | 6,9 |
| + CUDA top-k | 121 ms | 7,8 | 146 ms | 6,8 |
| + BM25 OpenMP | 94 ms | 9,7 | 144 ms | 6,8 |
| + FP16 | 45 ms | 19,7 | 57 ms | 16,5 |
| + dynamic batching | 50 ms | **30,7** | 60 ms | 15,4 |

**End-to-end dengan LLM, throughput token sistem (tok/s):**

| Pengguna | Baseline asli (Ollama) | Pipeline baseline di llama-server (serial) | Optimized |
|---:|---:|---:|---:|
| 1 | 13,5 | 16,5 | 16,3 |
| 10 | 13,1 | 16,0 | 32,3 |
| 50 | 13,6 | 15,8 | 32,5 |
| 100 | 13,0 | 15,0 | 28,7 |

Kolom tengah adalah ablation. Runtime baru (semua layer di GPU) menyumbang
±1,2×, dan continuous batching 4 permintaan menambah ±2×.

**Analisis bottleneck & hukum Amdahl:**
- **Retrieval 10⁶ chunk.**
  - Sebelum optimasi, waktu terbagi rata: vektor 38%, BM25 18%, rerank 38%.
  - CUDA dan OpenMP mempercepat dua tahap pertama 9–20×, tapi tanpa FP16
    percepatan keseluruhan tertahan di 2,0×. Inilah hukum Amdahl: rerank
    tidak tersentuh.
  - Setelah FP16, rerank kembali menjadi bottleneck (±70% waktu).
  - Dynamic batching tidak mempercepat satu permintaan, tapi menaikkan
    throughput 56% karena satu kali baca matriks di GPU dipakai banyak kueri.
- **KB resmi (1.920 chunk).**
  - Rerank memakan 97% waktu, jadi hanya FP16 yang berpengaruh (2,5×).
  - Batching tidak menambah throughput karena rerank sudah compute-bound
    (±130 ribu token/s), sesuai profil reranker.
- **End-to-end.**
  - Decode LLM adalah > 95% waktu, sehingga percepatan retrieval hampir tak
    terasa per jawaban.
  - Yang mengubah kapasitas adalah batching LLM: 2× jawaban per detik dan
    TTFT p50 pada 100 pengguna turun 86 → 61 s.
  - Antrean tetap panjang karena 4 slot × ±19 tok/s belum cukup untuk 10+
    pengguna. Ini bottleneck berikutnya: perlu GPU lebih besar atau model
    lebih kecil.
- **Build indeks.**
  - Setelah parsing paralel (±8×: 13,9 → 1,7 s dan 25,4 → 2,9 s), embedding menjadi 73–80% waktu build dan
    compute-bound di FP16.

**Analisis tingkat kernel** (`tools.bench_kernels`, `tools.profile_gpu`; grafik di `results/figures/`):
- **Kernel similarity CUDA memory-bound.**
  - FP32 pada N = 10⁵ mencapai 264 GB/s, setara bandwidth salin memori GPU
    (258 GB/s).
  - Batching 16 kueri menurunkan waktu per kueri 7× (10⁶: 7,1 → 1,0 ms), karena
    matriks dibaca sekali per batch.
  - Penyimpanan FP16 menghemat separuh VRAM, tetapi hanya 1,3× lebih cepat dari FP32
    (112 vs 169 GB/s pada 10⁶). Lane membaca `half2` 4 byte per instruksi, sehingga
    peluang berikutnya adalah load 8–16 byte per lane.
- **BM25 OpenMP**, strong scaling pada 65 juta posting:
  - speedup 1,9× (2 thread), 3,3× (4 thread), 5,2× (16 thread);
  - efisiensi turun dari 95% ke 32% karena beban memory-bound, CPU hibrida (6 P-core +
    4 E-core), dan penggabungan top-k yang serial.
- **Profil GPU satu batch** (16 kueri, 10⁶ chunk):
  - ±75% waktu GPU adalah reranker (GEMM tensor core FP16 + memory-efficient attention);
  - kernel similarity kita hanya ±3%;
  - konversi bobot FP32 → FP16 oleh autocast memakan ±3%. Ini biaya dari tidak
    menyimpan salinan bobot FP16 di VRAM yang sempit.

**Kualitas tidak berubah:**
- konteks top-5 identik untuk 20/20 pertanyaan;
- Recall@k dan MRR identik untuk 500 pertanyaan;
- hasil ingest paralel identik dengan sekuensial untuk 1.014 dokumen;
- embedding FP16 setara FP32 (cosine 1,000000).

**Keterbatasan pengukuran:**
- Laptop dalam kondisi mengisi daya dan GPU dibatasi daya (±34 W saat LLM).
  Angka baseline hari ini ±20–25% lebih lambat daripada 20 Nov, tetapi kedua
  mode diukur dalam kondisi yang sama.
- Aplikasi desktop memakai ±0,6–1,3 GB VRAM. Bila VRAM bebas kurang dari
  ±5,5 GB, `tools.llm_server` otomatis menaruh sebagian layer di CPU
  (`--fit`).
- Hasil yang dibatalkan atau terkena bias disimpan di `results/_arsip/` dan
  tidak dipakai: run 21 Nov yang terhenti, dan run LLM dengan cache prompt
  llama.cpp aktif.

## Data dummy, benchmarking & profiling

```bash
# Skala Tabel 3.1 (data sintetis, entitas fiktif, + QA ground truth di data/dummy/<koleksi>/)
.venv/bin/python -m tools.gen_dummy kb --collection dummy_s --chunks 10000 --index real      # embedding asli
.venv/bin/python -m tools.gen_dummy kb --collection dummy_m --chunks 100000 --index fake     # vektor acak: cepat, untuk latensi pencarian
.venv/bin/python -m tools.gen_dummy kb --collection dummy_l --chunks 1000000 --index fake
.venv/bin/python -m tools.gen_dummy files --out data/dummy/files_pdf_1000 --docs 1000 --format pdf   # untuk uji parsing
```

```bash
# Waktu pembangunan indeks (fase offline dari nol, rata-rata ± std)
.venv/bin/python -m tools.kb bench-build data/scrape/raw --repeats 3
.venv/bin/python -m tools.kb bench-build data/dummy/files_pdf_1000 --repeats 3
```

```bash
# Kualitas retrieval: Recall@k & MRR untuk vektor, BM25, hybrid, reranked
.venv/bin/python -m tools.eval_recall --collection dummy_s --qa data/dummy/dummy_s/qa.jsonl
```

```bash
# Uji otomatis kebenaran optimasi (±20 detik; -m "not slow" untuk melewati tes yang memuat model)
.venv/bin/python -m pytest
```

```bash
# Micro-benchmark kernel: bandwidth CUDA vs N & batch, strong scaling BM25 OpenMP 1..16 thread
.venv/bin/python -m tools.bench_kernels
```

```bash
# Profil GPU tingkat kernel (torch.profiler/CUPTI) + timeline Perfetto -> results/profiling/
.venv/bin/python -m tools.profile_gpu --collection dummy_l --mode optimized --batch 16
```

```bash
# Grafik untuk laporan dari semua hasil -> results/figures/fig1..fig8.png
.venv/bin/python -m tools.plots
```

```bash
# Uji beban: sapuan konkurensi, metrik Tabel 3.2 → results/loadtest/<tag>/summary.{md,csv,json}
.venv/bin/python -m loadtest.run --users 1 10 50 100 --duration 60 --repeats 3
.venv/bin/python -m loadtest.run --users 1 10 50 --no-generate --collection dummy_l --questions data/dummy/dummy_l/questions.txt
```

```bash
# Semua sekaligus, berurutan (±75 menit): retrieval ugm & dummy_l, build indeks, end-to-end LLM
bash loadtest/run_full.sh            # baseline
bash loadtest/run_full.sh optimized  # setelah modul C++ tersedia
```

Uji beban juga bisa dijalankan langsung dari UI (tab *Monitor & Uji Beban*)
sambil chat tetap dipakai, atau lewat web UI Locust
(`locust -f loadtest/locustfile.py --host http://127.0.0.1:8000`).

Di akhir setiap run, request yang masih antre setelah `--stop-timeout` dicatat
sebagai **Ditinggal**. Server mendeteksi klien yang putus lalu membatalkan
request tersebut, termasuk yang masih menunggu slot LLM, supaya run berikutnya
tidak ikut terbebani.

| Metrik (Tabel 3.2) | Sumber |
|---|---|
| Latensi per tahap | `timings_ms` per request (server) → `STAGE/*` di Locust, `/api/metrics`, JSONL |
| Latensi e2e p50/p95 | `METRIC/e2e` (klien Locust) |
| TTFT | `METRIC/ttft` (klien) + `ttft_ms` (server) |
| Throughput (QPS) | request selesai / waktu run sebenarnya (termasuk masa tunggu akhir) |
| Waktu pembangunan indeks | `tools.kb bench-build`, `index_meta.json` |
| Utilisasi CPU/GPU, VRAM | psutil + NVML disampling selama run |
| Kualitas (Recall@k) | `tools.eval_recall` |

### Hasil benchmark baseline (20 Nov 2026)

Mesin uji: RTX 5050 Laptop 8 GB, 16 core, RAM 15 GB, Ollama `qwen3.5:9b` Q4_K_M
(sebagian layer di CPU karena VRAM tidak cukup). Dijalankan dengan
`bash loadtest/run_full.sh baseline`: 3 pengulangan per titik, 60 detik per run
untuk retrieval dan 120 detik untuk end-to-end, jeda pengguna 1–3 detik.
Tidak ada request yang gagal di semua run. Angka lengkap (± std) ada di
`results/loadtest/*/summary.md` dan `results/index_build/*.json`.

**Retrieval saja**: embedding kueri → vektor + BM25 → RRF → rerank 20 kandidat.

| Pengguna | KB resmi (1.920 chunk): QPS | p50 / p95 (s) | 10⁶ chunk: QPS | p50 / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 0,48 | 0,13 / 0,15 | 0,45 | 0,19 / 0,22 |
| 10 | 4,5 | 0,17 / 0,42 | 4,2 | 0,32 / 0,80 |
| 50 | 6,6 | 5,5 / 5,9 | 5,3 | 7,3 / 7,8 |
| 100 | 6,5 | 13 / 13,7 | 5,3 | 17 / 17 |

Rincian per tahap pada 1 pengguna (ms):

| Tahap | KB resmi | 10⁶ chunk |
|---|---:|---:|
| embed_query | 4,7 | 4,7 |
| search_vector | 0,3 | 69 |
| search_bm25 | 0,1 | 36 |
| rerank | 117 | 73 |

- Pada 10⁶ chunk, pencarian vektor + BM25 (±105 ms) sudah melampaui rerank dan
  menjadi ±55% waktu layanan. Ini target utama CUDA top-k dan OpenMP BM25.
- Throughput jenuh di ±6,5 QPS (KB resmi) dan ±5,3 QPS (10⁶ chunk), karena
  baseline hanya punya satu slot retrieval. Di atas titik jenuh, latensi
  bertambah linear lewat `queue_retrieval`, yang sudah FIFO.

**End-to-end dengan LLM** (KB resmi):

| Pengguna | Selesai / run | Ditinggal / run | QPS | TTFT p50 (s) | E2E p50 (s) | `queue_llm` (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0 | 0,045 | 1,4 | 15,8 | 0 |
| 10 | 7 | 8 | 0,041 | 72 | 91 | 65 |
| 50 | 7 | 48 | 0,037 | 73 | 101 | 68 |
| 100 | 7 | 97 | 0,041 | 86 | 99 | 79 |

- LLM melayani satu permintaan pada satu waktu, jadi throughput tetap ±0,04 QPS
  (±2,4 jawaban/menit) berapa pun jumlah pengguna. Satu jawaban butuh ±15–25
  detik decode.
- Mulai 10 pengguna, hampir semua request masih antre saat run selesai
  (kolom *Ditinggal*). Latensi p50/p95 hanya dihitung dari request yang selesai,
  sehingga tertahan di batas jendela run (±180 detik); antrean sebenarnya terus
  bertambah.
- Retrieval < 1% dari latensi end-to-end. Saat LLM ikut berjalan di GPU, rerank
  melambat dari 120 menjadi ±168 ms. VRAM terpakai 7,4–7,7 dari 8 GB, RAM
  sistem maks 12,5 GB.

**Build indeks dari nol** (parse + clean + chunk + embedding batch 1 FP32 + BM25):

| Masukan | Chunk | Total | Parse + clean | Embedding |
|---|---:|---:|---:|---:|
| 14 dokumen resmi (`data/scrape/raw`) | 1.920 | 25,1 ± 0,8 s | 8,1 s (32%) | 16,8 s (67%) |
| 1.000 PDF sintetis | 6.547 | 52,7 ± 1,1 s | 16,7 s (32%) | 34,7 s (66%) |

**Kualitas retrieval** (dummy_s, 10⁴ chunk, 200 pertanyaan):

| Metode | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| Vektor | 0,90 | 0,99 | 0,94 |
| BM25 | 0,62 | 1,00 | 0,80 |
| Hybrid RRF | 0,91 | 1,00 | 0,95 |
| Hybrid + rerank | 0,98 | 1,00 | 0,99 |

**Grounding**: pertanyaan di luar KB ditolak dalam ±0,2 detik tanpa memanggil
LLM. Jawaban di dalam KB selalu menyertakan sitasi `[n]`.

## Catatan & troubleshooting

- **VRAM 8 GB sempit**: Ollama qwen3.5:9b (Q4) ±5,5 GB, ditambah embedder +
  reranker ±1 GB. Jika kurang, set `models.reranker.device: cpu` atau pakai
  model LLM lebih kecil di `config.local.yaml`.
- **llama-server**: `tools.llm_server` mematikan cache prompt di RAM host
  (`--cache-ram 0`). Default llama.cpp 8 GB sempat menghabiskan RAM, dan
  prompt yang ter-cache dari tahap uji sebelumnya membiaskan pengukuran.
  Jumlah layer di GPU menyesuaikan VRAM bebas (`--fit`). Tutup browser bila
  ingin semua 34 layer di GPU.
- **RAM 15 GB juga sempit saat benchmark**. Bagian model yang tidak muat di VRAM
  membuat Ollama memakai ±5,7 GB RAM, dan snapshot 10⁶ chunk di server ±2 GB.
  Tutup aplikasi berat (browser) saat menjalankan `run_full.sh`. Skrip ini
  melepas snapshot dummy_l setelah tahap 2 lewat
  `POST /api/kb/dummy_l/unload`. Benchmark 18 Nov sempat berhenti karena OOM
  (server membengkak ke 8,2 GB).
- **Retrieval berjalan di pool worker tetap** (`RetrievalPool` di
  `rag/service.py`), bukan di thread request. Ratusan thread request yang
  masing-masing menyentuh NumPy/SQLite membuat RSS bertambah mengikuti jumlah
  pengguna. `python -m server` juga membatasi arena malloc glibc.
- **Model GPU dipanggil lewat satu thread tetap per model** (`rag/models/worker.py`).
  Alasannya, PyTorch membuat cuBLAS handle + workspace VRAM untuk setiap thread
  pemanggil. Kalau model dipanggil langsung dari thread pool server, VRAM
  bertambah mengikuti jumlah pengguna sampai muncul `CUBLAS_STATUS_ALLOC_FAILED`
  (terjadi pada 50+ pengguna sebelum diperbaiki).
- **Server tidak bisa dihubungi dari UI**: pastikan `python -m server` berjalan
  dan `ui.api_url` di config benar.
- **Pertanyaan relevan malah ditolak**: turunkan `retrieval.min_rerank_score`
  (default 0,10), atau tambahkan dokumen resmi yang memuat jawabannya.
- Setiap pertanyaan dijawab independen; riwayat chat tidak dikirim ke LLM,
  supaya jawaban selalu bersumber dari dokumen.
