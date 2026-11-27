# Usulan isian bagian merah proposal (dari hasil implementasi & pengukuran)

Dokumen ini memetakan setiap placeholder merah di `Proposal_HPC_Revisi.pdf` ke angka
dan pilihan nyata proyek. Sumber angka: [results/comparison.md](../results/comparison.md),
`results/ablation/`, `results/kernels/`, dan README bagian *Hasil sebelum–sesudah*.

## 1.5 Batasan Masalah, butir 1 (korpus tambahan)

> [misalnya abstrak repositori skripsi/tesis atau jurnal UGM — pastikan ketersediaan dan izinnya]

Usulan: *"Untuk pengujian skalabilitas, korpus diperluas dengan **data sintetis**
(dokumen akademik fiktif berstruktur mirip peraturan, dibangkitkan dengan
`tools/gen_dummy.py`) hingga 10⁵ dan 10⁶ chunk, ditambah 1.000 PDF sintetis untuk uji
parsing. Data sintetis dipilih agar tidak bergantung pada izin repositori dan selalu
ditandai terpisah dari dokumen resmi."*

## 3.1 Skenario Penggunaan (target kinerja)

> [Tetapkan angka target setelah profiling baseline, misalnya p95 < X detik pada Y pengguna konkuren.]

Usulan target (baseline → hasil tercapai, 23 Nov 2026):

| Target | Baseline | Tercapai |
|---|---:|---:|
| Retrieval korpus 10⁶ chunk: **p95 < 1 detik pada 50 pengguna konkuren** | 11,0 s | 0,70 s |
| Retrieval KB resmi: **throughput ≥ 2× baseline** pada 50 pengguna | 6,3 QPS | 13,9 QPS (2,2×) |
| End-to-end dengan LLM: **throughput token ≥ 2× baseline** pada ≥ 10 pengguna | 13,1–13,6 tok/s | 32,3–32,5 tok/s (2,4×) |
| End-to-end, 1 pengguna: **TTFT p50 < 2 detik** | 1,4 s | 1,8 s |
| **Build indeks dokumen resmi < 15 detik** (pembaruan dokumen cepat) | 30,6 s | 9,6 s |

Catatan jujur untuk laporan: p95 end-to-end pada ≥ 10 pengguna tetap puluhan detik,
karena satu GPU 8 GB hanya cukup untuk 4 slot LLM paralel (±19 token/s per slot). Ini
batas perangkat keras, bukan batas pipeline.

## Tabel 3.1 (sumber korpus)

| Skala | Jumlah chunk nyata | Isi |
|---|---:|---|
| Kecil | 1.920 (resmi) + 10.002 (`dummy_s`) | 14 dokumen akademik resmi UGM + data sintetis |
| Sedang | 100.007 (`dummy_m`) | data sintetis (`tools/gen_dummy.py`) |
| Besar | 1.000.006 (`dummy_l`) | data sintetis (`tools/gen_dummy.py`) |

## Tabel 3.3 (perangkat keras & lunak)

| Komponen | Spesifikasi |
|---|---|
| CPU | Intel Core i5-13450HX — 10 core (6 P-core + 4 E-core), 16 thread |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, 20 SM, compute capability 12.0 (Blackwell) |
| RAM | 16 GB (15,3 GB terbaca) |
| Bahasa | Python 3.12 (orkestrasi); C++17 + OpenMP dan CUDA C++ (nvcc 12.0, GCC 12), pybind11 3.1 |
| Pustaka | PyTorch 2.15 (CUDA 13.2), Hugging Face Transformers 5.23, FastAPI 0.163, NumPy/SciPy |
| Model | embedding `intfloat/multilingual-e5-small` (d = 384); reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; LLM Qwen3.5 9B Q4_K_M (`qwen3.5:9b`), dilayani llama.cpp `llama-server` |
| Profiling | timer per tahap + `cudaEvent` (kernel C++), `torch.profiler` / CUPTI (`tools/profile_gpu.py`), micro-benchmark (`tools/bench_kernels.py`), ablation (`tools/ablation.py`) |
| Uji beban | Locust 2.47 |
| Antarmuka | Gradio 6.32 |

Catatan profiling: Nsight Systems/Compute versi paket Ubuntu (2022.4, CUDA 12.0) belum
mendukung GPU sm_120. `nsys` gagal membuat laporan karena komponen *importer* tidak ada,
dan `ncu` butuh izin performance counter (root). Bila tetap ingin memakai Nsight,
pasang Nsight Systems/Compute ≥ 2025.1 dari situs NVIDIA (butuh sudo), lalu aktifkan
izin counter (`NVreg_RestrictProfilingToAdminUsers=0`). Kalau tidak, ubah kalimat di
Bab 2.5/3.5 agar menyebut `torch.profiler` (CUPTI).

## Bagian lain yang perlu disesuaikan dengan implementasi

- **2.3 / 3.6 butir 4 (vLLM).** Yang dipakai adalah llama.cpp `llama-server` dengan
  continuous batching. vLLM tidak muat untuk model 9B di GPU 8 GB, dan Ollama menolak
  permintaan paralel untuk arsitektur qwen35.
- **3.6 butir 1 (parsing OpenMP).** Parsing diparalelkan dengan *pool proses*
  (penjadwalan dinamis LPT, PDF tebal dipecah per 32 halaman). Alasannya, parser
  PDF/HTML adalah pustaka Python yang tertahan GIL. OpenMP dipakai di BM25 (C++).
- **3.6 butir 2 (CUDA streams embedding).** Overlap dilakukan dengan thread produsen
  untuk tokenisasi dan *pinned memory* + `non_blocking` copy. Embedding memakai batch 64
  FP16 (autocast).
- **3.8 Demonstrasi.** Gradio, pilihan mode, tombol *Bandingkan mode*, panel latensi
  per tahap, uji beban langsung, dan halaman *Hasil Speedup* (ringkasan pengukuran
  sebelum–sesudah beserta grafiknya), dijalankan dengan `bash demo.sh`.
- **4.1 Jadwal.** "[Sesuaikan jumlah minggu dengan tenggat mata kuliah]": isi sesuai
  tenggat kelas.
- **Daftar pustaka [6].** "[Nama depan] Omri": cek nama depan penulis di *Parallel
  Computing* vol. 128 (2026).
