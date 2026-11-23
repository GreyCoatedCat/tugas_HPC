# Modul akselerasi C++ `rag_hpc` (CUDA + OpenMP)

Modul pybind11 ini menyediakan backend untuk dua tahap retrieval yang menjadi
bottleneck pada korpus besar (10⁶ chunk, lihat hasil baseline di README utama):
pencarian vektor dan BM25. Setelah dibangun, `rag/backends.py` memuatnya
otomatis lewat `rag/hpc.py`, dan mode `optimized` menjadi tersedia.

```bash
bash cpp/build.sh        # -> rag_hpc.cpython-312-x86_64-linux-gnu.so di root proyek
.venv/bin/python -c "import rag.backends as b; print(b.HPC_STATUS, list(b.VECTOR_BACKENDS), list(b.BM25_BACKENDS))"
```

Kebutuhan build: CMake ≥ 3.24, Ninja, nvcc (CUDA ≥ 12.0), GCC dengan OpenMP,
dan `pybind11` (ada di requirements.txt). nvcc 12.0 dari paket Ubuntu hanya
menerima host compiler GCC ≤ 12, sehingga `build.sh` memakai `g++-12`.

nvcc 12.0 juga belum mengenal arsitektur sm_120 (RTX 50xx). Karena itu kode
dikompilasi sebagai PTX `compute_90` dan di-JIT oleh driver saat pertama kali
dimuat. Hasil JIT di-cache driver, dan server memicunya saat warmup. Dengan
CUDA ≥ 12.8, kompilasi langsung bisa dipakai:
`bash cpp/build.sh -DCMAKE_CUDA_ARCHITECTURES=120`.

## Struktur

| Berkas | Isi |
|---|---|
| `src/topk_cuda.cu`, `.h` | kelas `CudaTopK`: matriks embedding di GPU (FP16), kernel similarity multi-kueri, radix-select top-k |
| `src/bm25_omp.cpp`, `.h` | kelas `OpenMPBM25`: akumulasi posting list paralel per segmen dokumen |
| `src/bindings.cpp` | binding pybind11, melepas GIL selama komputasi |
| `CMakeLists.txt`, `build.sh` | build: CUDA sebagai library statis, cudart statis |

## 1. Pencarian vektor di GPU (`cuda_topk`)

Embedding sudah dinormalisasi L2, sehingga cosine = dot product. Alur satu
batch berisi m kueri (m ≤ 16):

1. **Kernel similarity** (`scores_kernel`)
   - Satu warp memproses satu baris matriks. Lane-lane yang berurutan membaca
     elemen `half2` yang berurutan, sehingga akses global ter-coalesce (128 B
     per warp).
   - Ke-m kueri disimpan di *shared memory*. Setiap baris dibaca **sekali**
     lalu dikalikan dengan semua kueri, dengan akumulasi FP32 di register dan
     reduksi `__shfl_down_sync`.
   - Tahap ini *memory-bound*. Batching m kueri hampir tidak menambah waktu,
     karena biaya membaca matriks dibagi m.
2. **Radix select**
   - Skor float dipetakan ke kunci `uint32` yang monoton.
   - Empat putaran histogram 8-bit (atomik di shared memory, satu blok y per
     kueri) mempersempit prefiks kunci ke-k terbesar.
   - Seleksi digit berjalan di GPU, tanpa sinkronisasi host di antara putaran.
3. **Gather**: kompaksi paralel dengan dua penghitung atomik yang mengambil
   tepat k indeks (> ambang, lalu = ambang sebanyak yang masih dibutuhkan).
4. **Host**: k kandidat per kueri diurutkan. k kecil (50), jadi murah.

Matriks disalin ke GPU **sekali per snapshot indeks**. Konversi FP32 → FP16
dilakukan di GPU per potongan 64 MB. Penyimpanan FP16 menghemat separuh VRAM
dan bandwidth: 10⁶ × 384 → 768 MB.

Setiap instans memakai stream CUDA non-blocking sendiri dan dilindungi mutex.
Waktu kernel diukur dengan `cudaEvent` (`h2d_ms`, `score_ms`, `select_ms`,
`d2h_ms`) dan diteruskan ke panel latensi UI.

**Kebenaran.** Top-50 identik dengan `NumpyBruteForce` (overlap 1,0 pada 20
kueri × 10⁵ vektor, FP32 maupun FP16). Galat skor maksimal 5·10⁻⁵ (FP16).
Recall@k pada dummy_s identik dengan baseline.

## 2. BM25 paralel (`openmp`)

Matriks CSC berisi bobot BM25 final, dengan nomor dokumen terurut naik per
term. Rentang dokumen dibagi menjadi S = 4 × thread segmen
(`schedule(dynamic, 1)`). Untuk setiap term kueri, satu segmen:

- mencari batas segmennya di posting list dengan `lower_bound` (binary
  search);
- mengakumulasi bobot ke array skor bersama. Segmen saling lepas, jadi
  **tanpa atomik dan tanpa race**;
- mencatat dokumen yang tersentuh, lalu mengambil top-k lokal dengan
  `nth_element` dan mengembalikan akumulator ke nol.

Top-k lokal semua segmen kemudian digabung dengan `partial_sort`.

Pengukuran pada 10⁶ chunk (65 juta posting, 40 kueri):

| Thread | ms/kueri | vs scipy (35,3 ms) |
|---:|---:|---:|
| 1 | 9,7 | 3,6× |
| 2 | 6,2 | 5,7× |
| 4 | 3,4 | 10,5× |
| 8 | 2,4 | 15,0× |
| 16 | 2,0 | 18,0× |

Satu thread sudah 3,6× lebih cepat karena tidak ada `bincount` atas
seluruh 10⁶ dokumen dan seleksi berjalan hanya atas dokumen yang tersentuh.
Skalabilitas melandai setelah ±8 thread karena CPU i5-13450HX hanya punya 6
P-core + 4 E-core dan akses posting list bersifat memory-bound. Mode
teroptimasi memakai 8 thread.

Hasilnya identik dengan scipy sampai toleransi float (2·10⁻⁶). Satu-satunya
perbedaan adalah urutan dokumen yang skornya persis seri di batas top-k.

## 3. Pemakaian dari Python

```python
import rag_hpc, numpy as np
g = rag_hpc.CudaTopK(emb_f32, fp16=True)          # salin ke GPU sekali
idx, scores, t = g.search(queries_f32_mxd, k=50)   # [m,k], [m,k], {"score_ms":..}
b = rag_hpc.OpenMPBM25(csc.data, csc.indices, csc.indptr.astype(np.int64), n_docs, threads=8)
idx, scores, t = b.search(np.array(term_ids, np.int32), k=50)
rag_hpc.device_info()   # nama GPU, compute capability, VRAM
```

Wrapper yang memenuhi kontrak backend (`search`, `search_batch`) ada di
`rag/hpc.py`.
