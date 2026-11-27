// Kernel CUDA untuk pencarian vektor brute-force (cosine = dot product pada vektor ter-normalisasi).
//
// Alur per batch kueri (m <= max_batch):
//   1. scores_kernel  : satu warp per baris matriks; baris dibaca SEKALI lalu dikalikan dengan
//                       m kueri yang disimpan di shared memory. Tahap ini memory-bound, sehingga
//                       batching m kueri hampir tidak menambah waktu (biaya baca matriks dibagi m).
//   2. radix select   : mencari skor ke-k terbesar per kueri dengan 4 putaran histogram 8-bit
//                       (shared-memory atomics) atas kunci float yang dibuat bisa diurutkan
//                       sebagai uint32. Semua putaran berjalan di GPU tanpa sinkronisasi host.
//   3. gather         : kompaksi paralel — ambil tepat k indeks dengan skor >= ambang.
//   4. host           : urutkan k kandidat per kueri (k kecil, mis. 50).
#include "topk_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Saat gagal, status error per-thread dibersihkan (cudaGetLastError) sebelum melempar: error non-sticky
// seperti OOM pada cudaMalloc tidak boleh "terbaca ulang" oleh pengecekan peluncuran kernel berikutnya.
#define CUDA_CHECK(x)                                                                         \
    do {                                                                                      \
        cudaError_t err__ = (x);                                                              \
        if (err__ != cudaSuccess) {                                                           \
            (void)cudaGetLastError();                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__) + \
                                     " at " #x);                                              \
        }                                                                                     \
    } while (0)

namespace rag_hpc {
namespace {

constexpr int kMaxBatch = 16;     // batas register akumulator per warp
constexpr int kBlock = 256;
constexpr int kWarpsPerBlock = kBlock / 32;

__device__ __forceinline__ uint32_t float_to_key(float f) {
    // Pemetaan monoton: float lebih besar -> uint32 lebih besar (termasuk bilangan negatif).
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct HalfRow {
    __device__ __forceinline__ static float2 load(const void* base, int64_t row, int d2, int c) {
        return __half22float2(reinterpret_cast<const __half2*>(base)[row * d2 + c]);
    }
};
struct FloatRow {
    __device__ __forceinline__ static float2 load(const void* base, int64_t row, int d2, int c) {
        return reinterpret_cast<const float2*>(base)[row * d2 + c];
    }
};

template <typename Row>
__global__ void scores_kernel(const void* __restrict__ mat, int64_t n, int d, const float* __restrict__ q,
                              int m, float* __restrict__ scores) {
    extern __shared__ float sq[];  // [m, d]
    for (int i = threadIdx.x; i < m * d; i += blockDim.x) sq[i] = q[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int d2 = d >> 1;
    const int64_t stride = (int64_t)gridDim.x * kWarpsPerBlock;
    for (int64_t row = (int64_t)blockIdx.x * kWarpsPerBlock + warp; row < n; row += stride) {
        float acc[kMaxBatch];
#pragma unroll
        for (int j = 0; j < kMaxBatch; ++j) acc[j] = 0.f;
        // lane berurutan membaca elemen berurutan -> akses global ter-coalesce (128 B per warp)
        for (int c = lane; c < d2; c += 32) {
            const float2 v = Row::load(mat, row, d2, c);
#pragma unroll
            for (int j = 0; j < kMaxBatch; ++j) {
                if (j < m) {
                    const float2 w = reinterpret_cast<const float2*>(sq + j * d)[c];
                    acc[j] = fmaf(v.x, w.x, fmaf(v.y, w.y, acc[j]));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kMaxBatch; ++j) {
            if (j < m) {
                float a = acc[j];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
                if (lane == 0) scores[(int64_t)j * n + row] = a;
            }
        }
    }
}

__global__ void init_select_kernel(uint32_t* prefix, uint32_t* mask, int* krem, int* cnt, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        prefix[i] = 0u;
        mask[i] = 0u;
        krem[i] = k;
        cnt[2 * i] = 0;
        cnt[2 * i + 1] = 0;
    }
}

__global__ void radix_hist_kernel(const float* __restrict__ scores, int64_t n, const uint32_t* __restrict__ prefix,
                                  const uint32_t* __restrict__ mask, int shift, unsigned int* __restrict__ hist) {
    __shared__ unsigned int h[256];
    const int q = blockIdx.y;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0u;
    __syncthreads();
    const uint32_t p = prefix[q], mk = mask[q];
    const float* s = scores + (int64_t)q * n;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const uint32_t key = float_to_key(s[i]);
        if ((key & mk) == p) atomicAdd(&h[(key >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (h[i]) atomicAdd(&hist[q * 256 + i], h[i]);
}

__global__ void radix_select_kernel(const unsigned int* __restrict__ hist, uint32_t* prefix, uint32_t* mask,
                                    int* krem, int shift) {
    // Satu blok per kueri; 256 bin cukup dipindai serial oleh satu thread.
    if (threadIdx.x != 0) return;
    const int q = blockIdx.x;
    const unsigned int k = (unsigned int)krem[q];
    unsigned int cum = 0;
    for (int b = 255; b >= 0; --b) {
        const unsigned int c = hist[q * 256 + b];
        if (cum + c >= k) {
            prefix[q] |= ((uint32_t)b << shift);
            mask[q] |= (0xFFu << shift);
            krem[q] = (int)(k - cum);  // berapa elemen dengan kunci = prefix yang masih dibutuhkan
            return;
        }
        cum += c;
    }
}

__global__ void gather_kernel(const float* __restrict__ scores, int64_t n, const uint32_t* __restrict__ kth,
                              const int* __restrict__ krem, int k, int* cnt, int64_t* __restrict__ out_idx,
                              float* __restrict__ out_s) {
    const int q = blockIdx.y;
    const uint32_t t = kth[q];
    const int need_eq = krem[q];
    const int n_gt = k - need_eq;
    const float* s = scores + (int64_t)q * n;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const float v = s[i];
        const uint32_t key = float_to_key(v);
        int pos = -1;
        if (key > t) {
            pos = atomicAdd(&cnt[2 * q], 1);
        } else if (key == t) {
            const int e = atomicAdd(&cnt[2 * q + 1], 1);
            if (e < need_eq) pos = n_gt + e;
        }
        if (pos >= 0) {
            out_idx[(int64_t)q * k + pos] = i;
            out_s[(int64_t)q * k + pos] = v;
        }
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t count) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x)
        dst[i] = __float2half_rn(src[i]);
}

template <typename T>
void free_ptr(T*& p) {
    if (p) cudaFree(p);
    p = nullptr;
}

}  // namespace

CudaTopK::CudaTopK(const float* emb, int64_t n, int d, bool fp16, int device)
    : n_(n), d_(d), fp16_(fp16), device_(device) {
    if (d <= 0 || (d % 2) != 0) throw std::invalid_argument("dimensi vektor harus genap dan > 0");
    // Destruktor tidak dipanggil bila konstruktor melempar: lepaskan sendiri yang sudah dialokasikan.
    try {
        init(emb, fp16);
    } catch (...) {
        release();
        throw;
    }
}

void CudaTopK::init(const float* emb, bool fp16) {
    const int64_t n = n_;
    const int d = d_;
    CUDA_CHECK(cudaSetDevice(device_));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    sm_count_ = prop.multiProcessorCount;
    // shared memory statis 48 KB per blok: m * d * 4 byte
    max_batch_ = std::max(1, std::min(kMaxBatch, int((48 * 1024) / (d * (int)sizeof(float)))));

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    stream_ = st;
    for (auto& e : ev_) {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreate(&ev));
        e = ev;
    }

    const size_t count = (size_t)n * (size_t)d;
    matrix_bytes_ = count * (fp16 ? sizeof(__half) : sizeof(float));
    if (n == 0) return;
    CUDA_CHECK(cudaMalloc(&d_matrix_, matrix_bytes_));
    if (!fp16) {
        CUDA_CHECK(cudaMemcpy(d_matrix_, emb, matrix_bytes_, cudaMemcpyHostToDevice));
        return;
    }
    // Konversi FP32 -> FP16 di GPU per potongan agar tidak perlu salinan half di host.
    const size_t chunk = std::min<size_t>(count, (size_t)16 << 20);  // 16 M elemen = 64 MB staging
    float* staging = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, chunk * sizeof(float)));
    try {
        for (size_t off = 0; off < count; off += chunk) {
            const size_t len = std::min(chunk, count - off);
            CUDA_CHECK(cudaMemcpyAsync(staging, emb + off, len * sizeof(float), cudaMemcpyHostToDevice, st));
            const int grid = (int)std::min<size_t>((len + kBlock - 1) / kBlock, (size_t)sm_count_ * 32);
            f32_to_f16_kernel<<<grid, kBlock, 0, st>>>(staging, static_cast<__half*>(d_matrix_) + off, (int64_t)len);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(st));
        }
    } catch (...) {
        cudaFree(staging);
        throw;
    }
    cudaFree(staging);
}

CudaTopK::~CudaTopK() { release(); }

void CudaTopK::release() noexcept {
    cudaSetDevice(device_);
    if (stream_) cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    free_ptr(d_matrix_);
    free_ptr(d_queries_);
    free_ptr(d_scores_);
    free_ptr(d_hist_);
    free_ptr(d_prefix_);
    free_ptr(d_mask_);
    free_ptr(d_krem_);
    free_ptr(d_cnt_);
    free_ptr(d_out_idx_);
    free_ptr(d_out_s_);
    for (auto& e : ev_)
        if (e) cudaEventDestroy(static_cast<cudaEvent_t>(e));
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
    for (auto& e : ev_) e = nullptr;
    cap_m_ = cap_k_ = 0;
}

void CudaTopK::ensure_capacity(int m, int k) {
    if (m > cap_m_) {
        cap_m_ = cap_k_ = 0;  // bila alokasi di bawah gagal, panggilan berikutnya mengalokasikan ulang
        free_ptr(d_queries_);
        free_ptr(d_scores_);
        free_ptr(d_hist_);
        free_ptr(d_prefix_);
        free_ptr(d_mask_);
        free_ptr(d_krem_);
        free_ptr(d_cnt_);
        CUDA_CHECK(cudaMalloc(&d_queries_, (size_t)m * d_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scores_, (size_t)m * n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hist_, (size_t)m * 256 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_prefix_, (size_t)m * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_mask_, (size_t)m * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_krem_, (size_t)m * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cnt_, (size_t)m * 2 * sizeof(int)));
        free_ptr(d_out_idx_);
        free_ptr(d_out_s_);
        cap_m_ = m;
    }
    if (k > cap_k_) {
        cap_k_ = 0;
        free_ptr(d_out_idx_);
        free_ptr(d_out_s_);
        CUDA_CHECK(cudaMalloc(&d_out_idx_, (size_t)cap_m_ * k * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_out_s_, (size_t)cap_m_ * k * sizeof(float)));
        cap_k_ = k;
    }
}

void CudaTopK::search(const float* queries, int m, int k, int64_t* out_idx, float* out_scores, CudaTimings* t) {
    if (m <= 0 || n_ == 0 || k <= 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    CUDA_CHECK(cudaSetDevice(device_));
    const int k_eff = (int)std::min<int64_t>(k, n_);
    for (int start = 0; start < m; start += max_batch_) {
        const int mm = std::min(max_batch_, m - start);
        search_chunk(queries + (size_t)start * d_, mm, k_eff, out_idx + (size_t)start * k_eff,
                     out_scores + (size_t)start * k_eff, t);
    }
}

void CudaTopK::search_chunk(const float* queries, int m, int k, int64_t* out_idx, float* out_scores, CudaTimings* t) {
    ensure_capacity(std::max(m, cap_m_), k);
    cudaStream_t st = static_cast<cudaStream_t>(stream_);
    cudaEvent_t e0 = static_cast<cudaEvent_t>(ev_[0]), e1 = static_cast<cudaEvent_t>(ev_[1]),
                e2 = static_cast<cudaEvent_t>(ev_[2]), e3 = static_cast<cudaEvent_t>(ev_[3]),
                e4 = static_cast<cudaEvent_t>(ev_[4]);

    CUDA_CHECK(cudaEventRecord(e0, st));
    CUDA_CHECK(cudaMemcpyAsync(d_queries_, queries, (size_t)m * d_ * sizeof(float), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaEventRecord(e1, st));

    const int score_grid = (int)std::min<int64_t>((n_ + kWarpsPerBlock - 1) / kWarpsPerBlock, (int64_t)sm_count_ * 16);
    const size_t smem = (size_t)m * d_ * sizeof(float);
    if (fp16_)
        scores_kernel<HalfRow><<<score_grid, kBlock, smem, st>>>(d_matrix_, n_, d_, d_queries_, m, d_scores_);
    else
        scores_kernel<FloatRow><<<score_grid, kBlock, smem, st>>>(d_matrix_, n_, d_, d_queries_, m, d_scores_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2, st));

    init_select_kernel<<<(m + 31) / 32, 32, 0, st>>>(d_prefix_, d_mask_, d_krem_, d_cnt_, m, k);
    CUDA_CHECK(cudaGetLastError());
    const int gx = (int)std::min<int64_t>((n_ + kBlock - 1) / kBlock, (int64_t)sm_count_ * 8);
    const dim3 grid(gx, m);
    for (int shift = 24; shift >= 0; shift -= 8) {
        CUDA_CHECK(cudaMemsetAsync(d_hist_, 0, (size_t)m * 256 * sizeof(unsigned int), st));
        radix_hist_kernel<<<grid, kBlock, 0, st>>>(d_scores_, n_, d_prefix_, d_mask_, shift, d_hist_);
        radix_select_kernel<<<m, 32, 0, st>>>(d_hist_, d_prefix_, d_mask_, d_krem_, shift);
    }
    gather_kernel<<<grid, kBlock, 0, st>>>(d_scores_, n_, d_prefix_, d_krem_, k, d_cnt_, d_out_idx_, d_out_s_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e3, st));

    CUDA_CHECK(cudaMemcpyAsync(out_idx, d_out_idx_, (size_t)m * k * sizeof(int64_t), cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(out_scores, d_out_s_, (size_t)m * k * sizeof(float), cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaEventRecord(e4, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    if (t) {
        float ms;
        cudaEventElapsedTime(&ms, e0, e1); t->h2d_ms += ms;
        cudaEventElapsedTime(&ms, e1, e2); t->score_ms += ms;
        cudaEventElapsedTime(&ms, e2, e3); t->select_ms += ms;
        cudaEventElapsedTime(&ms, e3, e4); t->d2h_ms += ms;
    }

    // Urutkan k kandidat per kueri (skor menurun, indeks menaik bila seri -> deterministik).
    std::vector<int> order(k);
    std::vector<int64_t> tmp_i(k);
    std::vector<float> tmp_s(k);
    for (int q = 0; q < m; ++q) {
        int64_t* ii = out_idx + (size_t)q * k;
        float* ss = out_scores + (size_t)q * k;
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return ss[a] > ss[b] || (ss[a] == ss[b] && ii[a] < ii[b]);
        });
        for (int r = 0; r < k; ++r) {
            tmp_i[r] = ii[order[r]];
            tmp_s[r] = ss[order[r]];
        }
        std::copy(tmp_i.begin(), tmp_i.end(), ii);
        std::copy(tmp_s.begin(), tmp_s.end(), ss);
    }
}

DeviceInfo device_info(int device) {
    DeviceInfo info{};
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::snprintf(info.name, sizeof(info.name), "%s", prop.name);
    info.major = prop.major;
    info.minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(&info.free_mem, &info.total_mem));
    cudaRuntimeGetVersion(&info.runtime_version);
    cudaDriverGetVersion(&info.driver_version);
    return info;
}

}  // namespace rag_hpc
