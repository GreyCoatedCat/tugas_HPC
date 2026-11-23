// Pencarian vektor brute-force di GPU: similarity (dot product) + seleksi top-k.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rag_hpc {

struct CudaTimings {
    float h2d_ms = 0.f;     // salin kueri host -> device
    float score_ms = 0.f;   // kernel similarity (memory-bound, satu kali baca matriks per batch)
    float select_ms = 0.f;  // radix select top-k + kompaksi kandidat
    float d2h_ms = 0.f;     // salin kandidat device -> host
};

class CudaTopK {
public:
    // emb: float32 [n, d] row-major, baris sudah dinormalisasi L2.
    // fp16=true menyimpan matriks sebagai half (hemat VRAM & bandwidth 2x), akumulasi tetap FP32.
    CudaTopK(const float* emb, int64_t n, int d, bool fp16, int device);
    ~CudaTopK();
    CudaTopK(const CudaTopK&) = delete;
    CudaTopK& operator=(const CudaTopK&) = delete;

    // queries: float32 [m, d]. Hasil per kueri terurut menurun:
    // out_idx/out_scores berukuran [m, k_eff] dengan k_eff = min(k, n).
    void search(const float* queries, int m, int k, int64_t* out_idx, float* out_scores, CudaTimings* t);

    int64_t size() const { return n_; }
    int dim() const { return d_; }
    bool fp16() const { return fp16_; }
    size_t matrix_bytes() const { return matrix_bytes_; }
    int max_batch() const { return max_batch_; }

private:
    void ensure_capacity(int m, int k);
    void search_chunk(const float* queries, int m, int k, int64_t* out_idx, float* out_scores, CudaTimings* t);

    int64_t n_ = 0;
    int d_ = 0;
    bool fp16_ = true;
    int device_ = 0;
    int sm_count_ = 1;
    int max_batch_ = 1;
    size_t matrix_bytes_ = 0;

    void* d_matrix_ = nullptr;          // __half* atau float* [n, d]
    float* d_queries_ = nullptr;        // [cap_m, d]
    float* d_scores_ = nullptr;         // [cap_m, n]
    unsigned int* d_hist_ = nullptr;    // [cap_m, 256]
    uint32_t* d_prefix_ = nullptr;      // [cap_m]
    uint32_t* d_mask_ = nullptr;        // [cap_m]
    int* d_krem_ = nullptr;             // [cap_m]
    int* d_cnt_ = nullptr;              // [cap_m, 2]
    int64_t* d_out_idx_ = nullptr;      // [cap_m, cap_k]
    float* d_out_s_ = nullptr;          // [cap_m, cap_k]
    int cap_m_ = 0;
    int cap_k_ = 0;

    void* stream_ = nullptr;            // cudaStream_t
    void* ev_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    std::mutex mu_;
};

// Info perangkat untuk laporan ("RTX 5050 Laptop GPU, sm_120, 20 SM").
struct DeviceInfo {
    char name[256];
    int major, minor, sm_count;
    size_t total_mem, free_mem;
    int runtime_version, driver_version;
};
DeviceInfo device_info(int device);

}  // namespace rag_hpc
