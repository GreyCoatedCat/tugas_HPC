# Hasil uji beban — 2026-11-20 11:44

- Mode: `baseline` | Koleksi: `ugm` | Generasi LLM: tidak (retrieval saja)
- Durasi per run: 60s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 ± 2 | 0 | 0 | 0.476 ± 0.035 | 127 ± 6 | 150 ± 10 | 127 ± 6 | 150 ± 10 | 7.6 ± 0.1 | 4.6 ± 3.3 | 2425 | 6.5 |
| 10 | 272 ± 8 | 0 | 0 | 4.517 ± 0.121 | 167 ± 12 | 420 ± 20 | 167 ± 12 | 420 ± 20 | 11.8 ± 0.1 | 51.7 ± 3.4 | 2319 | 6.4 |
| 50 | 434 ± 2 | 0 | 0 | 6.607 ± 0.039 | 5467 ± 58 | 5933 ± 58 | 5467 ± 58 | 5933 ± 58 | 14.5 | 89.1 ± 0.7 | 2319 | 6.4 |
| 100 | 478 ± 3 | 0 | 0 | 6.543 ± 0.051 | 13000 | 13667 ± 577 | 13000 | 13667 ± 577 | 14.6 ± 0.1 | 90.0 ± 1.1 | 2319 | 6.4 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 4.4 | 4.8 | 5.3 | 5.5 |
| fetch_chunks | 1.8 | 0.2 | 0.2 | 0.2 |
| fusion | 0.1 | 0.1 | 0.1 | 0.1 |
| queue_retrieval | 0.1 | 80.8 | 5133.2 | 11751.7 |
| rerank | 117.4 | 130.4 | 143.4 | 143.9 |
| search_bm25 | 0.1 | 0.1 | 0.1 | 0.1 |
| search_vector | 0.3 | 0.3 | 0.3 | 0.3 |
| server_e2e | 125.1 | 217.7 | 5283.9 | 11902.9 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 30s); server membatalkannya agar tidak membebani run berikutnya.
