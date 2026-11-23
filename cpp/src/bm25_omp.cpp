// BM25 paralel dengan OpenMP: dekomposisi data per segmen dokumen.
//
// Matriks CSC menyimpan posting list per term dengan nomor dokumen terurut naik. Rentang dokumen
// [0, n) dibagi menjadi S segmen (S = 4 x jumlah thread, dijadwalkan dinamis untuk menyeimbangkan
// beban). Untuk setiap term kueri, thread mencari batas segmennya di posting list dengan binary
// search, lalu mengakumulasi bobot ke array skor bersama. Segmen saling lepas -> tidak ada race
// dan tidak perlu atomik. Setiap segmen menghasilkan top-k lokal; hasil akhir = gabungan top-k lokal.
#include "bm25_omp.h"

#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rag_hpc {

OpenMPBM25::OpenMPBM25(const float* data, const int32_t* indices, const int64_t* indptr, int64_t n_docs,
                       int64_t n_terms, int threads)
    : data_(data), indices_(indices), indptr_(indptr), n_docs_(n_docs), n_terms_(n_terms),
      threads_(threads > 0 ? threads : omp_get_max_threads()), acc_((size_t)std::max<int64_t>(n_docs, 0), 0.f) {
    if (n_docs > INT32_MAX) throw std::invalid_argument("n_docs melebihi batas int32 indeks CSC");
}

int OpenMPBM25::search(const int32_t* cols, int ncols, int k, int64_t* out_idx, float* out_scores, BM25Timings* t) {
    if (n_docs_ == 0 || ncols <= 0 || k <= 0) return 0;
    std::lock_guard<std::mutex> lock(mu_);  // acc_ dipakai bersama -> satu kueri per instans pada satu waktu

    const int nseg = (int)std::max<int64_t>(1, std::min<int64_t>((int64_t)threads_ * 4, n_docs_));
    std::vector<std::vector<std::pair<float, int32_t>>> local((size_t)nseg);
    float* acc = acc_.data();

    const double t0 = omp_get_wtime();
#pragma omp parallel for num_threads(threads_) schedule(dynamic, 1)
    for (int s = 0; s < nseg; ++s) {
        const int32_t lo = (int32_t)(n_docs_ * s / nseg);
        const int32_t hi = (int32_t)(n_docs_ * (s + 1) / nseg);
        std::vector<int32_t> touched;
        touched.reserve(256);
        for (int ci = 0; ci < ncols; ++ci) {
            const int32_t c = cols[ci];
            if (c < 0 || c >= n_terms_) continue;
            const int32_t* begin = indices_ + indptr_[c];
            const int32_t* end = indices_ + indptr_[c + 1];
            const int32_t* a = std::lower_bound(begin, end, lo);
            const int32_t* z = std::lower_bound(a, end, hi);
            const float* w = data_ + (a - indices_);
            for (const int32_t* p = a; p < z; ++p, ++w) {
                const int32_t d = *p;
                if (acc[d] == 0.f) touched.push_back(d);  // bobot BM25 selalu > 0
                acc[d] += *w;
            }
        }
        auto by_score = [acc](int32_t x, int32_t y) { return acc[x] > acc[y] || (acc[x] == acc[y] && x < y); };
        const size_t take = std::min<size_t>((size_t)k, touched.size());
        if (touched.size() > take) std::nth_element(touched.begin(), touched.begin() + take, touched.end(), by_score);
        auto& L = local[(size_t)s];
        L.reserve(take);
        for (size_t i = 0; i < take; ++i) L.emplace_back(acc[touched[i]], touched[i]);
        for (int32_t d : touched) acc[d] = 0.f;  // kembalikan akumulator ke nol untuk kueri berikutnya
    }
    const double t1 = omp_get_wtime();

    std::vector<std::pair<float, int32_t>> all;
    for (auto& L : local) all.insert(all.end(), L.begin(), L.end());
    const size_t take = std::min<size_t>((size_t)k, all.size());
    std::partial_sort(all.begin(), all.begin() + take, all.end(), [](const auto& x, const auto& y) {
        return x.first > y.first || (x.first == y.first && x.second < y.second);
    });
    for (size_t i = 0; i < take; ++i) {
        out_idx[i] = all[i].second;
        out_scores[i] = all[i].first;
    }
    const double t2 = omp_get_wtime();
    if (t) {
        t->parallel_ms += (float)((t1 - t0) * 1e3);
        t->merge_ms += (float)((t2 - t1) * 1e3);
    }
    return (int)take;
}

}  // namespace rag_hpc
