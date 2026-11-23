// BM25 paralel (OpenMP) di atas matriks CSC berisi bobot BM25 final.
#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace rag_hpc {

struct BM25Timings {
    float parallel_ms = 0.f;  // akumulasi posting list + top-k lokal per segmen dokumen (paralel OpenMP)
    float merge_ms = 0.f;     // penggabungan kandidat semua segmen (serial, kecil)
};

class OpenMPBM25 {
public:
    // data/indices/indptr = array CSC [n_docs x V] (indices terurut naik per kolom).
    // Pointer tidak disalin; pemanggil (binding Python) menjaga array tetap hidup.
    OpenMPBM25(const float* data, const int32_t* indices, const int64_t* indptr, int64_t n_docs, int64_t n_terms,
               int threads);

    // cols: id term unik kueri. Hasil terurut menurun, hanya skor > 0; return jumlah hasil (<= k).
    int search(const int32_t* cols, int ncols, int k, int64_t* out_idx, float* out_scores, BM25Timings* t);

    int threads() const { return threads_; }
    int64_t n_docs() const { return n_docs_; }

private:
    const float* data_;
    const int32_t* indices_;
    const int64_t* indptr_;
    int64_t n_docs_;
    int64_t n_terms_;
    int threads_;
    std::vector<float> acc_;  // akumulator skor [n_docs]; tiap thread hanya menyentuh segmennya sendiri
    std::mutex mu_;
};

}  // namespace rag_hpc
