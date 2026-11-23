# Perbandingan sebelum vs sesudah optimasi — 2026-11-23 17:15

Sumber: hasil terbaru di `results/` (rata-rata 3 pengulangan per titik). Baseline dan optimized diukur pada sesi yang sama dengan protokol yang sama (`loadtest/run_full.sh`).

## Retrieval saja — KB resmi UGM (1.920 chunk)

`20261123-140354_baseline_ugm_retrieval` vs `20261123-143749_optimized_ugm_retrieval`

| Pengguna | QPS baseline | QPS optimized | Throughput | p50 baseline (ms) | p50 optimized (ms) | p95 baseline (ms) | p95 optimized (ms) | p95 lebih cepat |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.48 | 0.49 | 1.0× | 140 | 79 | 153 | 88 | 1.7× |
| 10 | 4.40 | 4.66 | 1.1× | 197 | 82 | 530 | 263 | 2.0× |
| 50 | 6.33 | 13.94 | 2.2× | 5,733 | 1,533 | 6,200 | 2,267 | 2.7× |
| 100 | 6.28 | 13.56 | 2.2× | 14,000 | 5,233 | 14,000 | 5,967 | 2.3× |

Rata-rata per tahap pada 1 pengguna (ms):

| Tahap | Baseline | Optimized | Percepatan |
|---|---:|---:|---:|
| embed_query | 8.6 | 11.9 | 0.7× |
| fetch_chunks | 0.3 | 0.3 | 0.8× |
| fusion | 0.1 | 0.2 | 0.7× |
| queue_retrieval | 0.1 | 0.2 | 0.8× |
| rerank | 123.8 | 59.3 | 2.1× |
| search_bm25 | 0.1 | 0.3 | 0.5× |
| search_vector | 0.4 | 0.4 | 1.2× |

## Retrieval saja — Korpus sintetis 10⁶ chunk

`20261123-141758_baseline_dummy_l_retrieval` vs `20261123-145112_optimized_dummy_l_retrieval`

| Pengguna | QPS baseline | QPS optimized | Throughput | p50 baseline (ms) | p50 optimized (ms) | p95 baseline (ms) | p95 optimized (ms) | p95 lebih cepat |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.44 | 0.45 | 1.0× | 257 | 60 | 297 | 70 | 4.3× |
| 10 | 3.75 | 4.75 | 1.3× | 620 | 61 | 1,200 | 147 | 8.2× |
| 50 | 3.93 | 20.99 | 5.3× | 10,333 | 343 | 11,000 | 697 | 15.8× |
| 100 | 3.92 | 26.59 | 6.8× | 23,000 | 1,733 | 24,000 | 2,100 | 11.4× |

Rata-rata per tahap pada 1 pengguna (ms):

| Tahap | Baseline | Optimized | Percepatan |
|---|---:|---:|---:|
| embed_query | 9.6 | 12.1 | 0.8× |
| fetch_chunks | 4.0 | 0.4 | 8.9× |
| fusion | 0.2 | 0.2 | 0.7× |
| queue_retrieval | 0.1 | 0.1 | 1.1× |
| rerank | 79.2 | 35.3 | 2.2× |
| search_bm25 | 52.2 | 5.9 | 8.9× |
| search_vector | 109.1 | 5.3 | 20.5× |

## End-to-end dengan LLM — KB resmi UGM

- *Baseline asli*: Ollama bawaan (25/34 layer di GPU, satu permintaan per waktu, parameter sampling bawaan model: `top_k 20`, `presence_penalty 1.5`).
- *Baseline (llama-server, serial)*: pipeline baseline di atas runtime LLM teroptimasi — ablation untuk memisahkan efek runtime dari efek batching. Parameter sampling bawaan llama.cpp menghasilkan jawaban lebih panjang, jadi bandingkan dengan baseline asli memakai **throughput token**.
- *Optimized*: pipeline teroptimasi (retrieval CUDA/OpenMP/FP16/batching + 4 permintaan LLM sekaligus) di runtime yang sama dengan baris tengah.

**Throughput token** (token jawaban yang dihasilkan sistem per detik — tidak bergantung panjang jawaban):

| Pengguna | Baseline asli (Ollama bawaan) | Baseline (llama-server, serial) | Optimized |
|---:|---:|---:|---:|
| 1 | 13.5 | 16.5 | 16.3 |
| 10 | 13.1 | 16.0 | 32.3 |
| 50 | 13.6 | 15.8 | 32.5 |
| 100 | 13.0 | 15.0 | 28.7 |

**Jawaban selesai per detik (QPS)** dan jumlah selesai / ditinggal per run:

| Pengguna | Baseline asli (Ollama bawaan) | Baseline (llama-server, serial) | Optimized |
|---:|---:|---:|---:|
| 1 | 0.045 (6 / 0) | 0.027 (4 / 0) | 0.039 (5 / 0) |
| 10 | 0.041 (7 / 8) | 0.041 (7 / 7) | 0.080 (14 / 5) |
| 50 | 0.037 (7 / 48) | 0.044 (8 / 47) | 0.074 (13 / 44) |
| 100 | 0.041 (7 / 97) | 0.037 (7 / 97) | 0.081 (15 / 95) |

**TTFT p50 / E2E p50 (detik)** — hanya dari permintaan yang selesai:

| Pengguna | Baseline asli (Ollama bawaan) | Baseline (llama-server, serial) | Optimized |
|---:|---:|---:|---:|
| 1 | 1.4 / 15.8 | 1.5 / 27.3 | 1.8 / 17.3 |
| 10 | 72.0 / 91.3 | 90.0 / 112.7 | 53.7 / 92.3 |
| 50 | 72.7 / 101.3 | 73.7 / 98.3 | 58.3 / 116.0 |
| 100 | 85.7 / 98.7 | 87.3 / 105.3 | 61.3 / 100.7 |

**Median token per jawaban:** Baseline asli (Ollama bawaan): 297 · Baseline (llama-server, serial): 434 · Optimized: 390

## Waktu build indeks dari nol

| Masukan | Baseline | Optimized | Percepatan | Ingest (parse+clean+chunk) | Embedding |
|---|---:|---:|---:|---:|---:|
| 14 dokumen resmi | 30.6 s | 9.6 s | 3.2× | 13.9 → 1.7 s | 16.5 → 7.7 s |
| 1.000 PDF sintetis | 64.0 s | 11.7 s | 5.5× | 25.4 → 2.9 s | 38.2 → 8.6 s |

## Server LLM (tools.llm_probe, prompt RAG ±2.000 token)

| Konfigurasi | Decode 1 permintaan (tok/s) | Agregat 4 permintaan (tok/s) | Waktu 4 jawaban (s) |
|---|---:|---:|---:|
| llamacpp-np4-fullgpu | 24.3 | 33.7 | 34.3 |
| llamacpp-np4-nocache-20261123 | 19.0 | 37.2 | 48.6 |
| ollama-default | 17.2 | 14.8 | 87.1 |
| ollama-par4-fa-q8 | 18.6 | 16.2 | 78.7 |

## Ablation per optimasi

Lihat `results/ablation/ablation-20261123-140352.md`.

