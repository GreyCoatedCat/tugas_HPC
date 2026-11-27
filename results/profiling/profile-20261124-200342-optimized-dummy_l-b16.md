# Profil GPU — mode `optimized`, koleksi `dummy_l`, batch 16 — 2026-11-24 20:03

3 batch direkam, waktu dinding 1837 ms (612.4 ms per batch, 38.3 ms per pertanyaan). Timeline: `trace-20261124-200342-optimized-dummy_l-b16.json` (buka di https://ui.perfetto.dev).

## Per tahap (CPU = waktu dinding tahap)

| Tahap | CPU total (ms) | GPU di dalam tahap (ms) |
|---|---:|---:|
| 1_embed_query | 31.5 | 0.0 |
| 2_search_vector | 40.6 | 0.0 |
| 2_search_vector | 0.0 | 39.8 |
| 3_search_bm25 | 133.3 | 0.0 |
| 4_fusion_fetch | 159.1 | 0.0 |
| 5_rerank | 1383.0 | 0.0 |

## 20 kernel/operator GPU terbesar

| Kernel / operator | Panggilan | Waktu GPU total (ms) | % |
|---|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align8...` | 936 | 332.59 | 25.2% |
| `fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass...` | 216 | 269.98 | 20.4% |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8>...` | 216 | 195.78 | 14.8% |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align...` | 120 | 145.95 | 11.1% |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, fals...` | 450 | 89.87 | 6.8% |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, std::a...` | 432 | 69.52 | 5.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::converting_copy_kernel_cu...` | 3522 | 45.25 | 3.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::Te...` | 216 | 43.58 | 3.3% |
| `void rag_hpc::(anonymous namespace)::scores_kernel<rag_hpc::(anonymous namespace)::Half...` | 3 | 34.86 | 2.6% |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align...` | 24 | 18.96 | 1.4% |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 216 | 10.86 | 0.8% |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 207 | 7.14 | 0.5% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, s...` | 33 | 3.44 | 0.3% |
| `rag_hpc::(anonymous namespace)::radix_hist_kernel(float const*, long, unsigned int cons...` | 12 | 3.36 | 0.3% |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, lon...` | 54 | 2.70 | 0.2% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::Half>, s...` | 624 | 2.54 | 0.2% |
| `rag_hpc::(anonymous namespace)::gather_kernel(float const*, long, unsigned int const*, ...` | 3 | 1.13 | 0.1% |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 36 | 0.79 | 0.1% |
| `Memcpy HtoD (Pageable -> Device)` | 39 | 0.42 | 0.0% |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_tn_align8...` | 15 | 0.32 | 0.0% |

