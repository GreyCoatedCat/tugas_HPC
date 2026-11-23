# Hasil uji beban — 2026-11-20 11:58

- Mode: `baseline` | Koleksi: `dummy_l` | Generasi LLM: tidak (retrieval saja)
- Durasi per run: 60s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 ± 1 | 0 | 0 | 0.454 ± 0.019 | 190 | 220 ± 10 | 190 | 220 ± 10 | 7.5 ± 0.2 | 4.0 ± 1.5 | 2319 | 8.4 |
| 10 | 253 ± 1 | 0 | 0 | 4.185 ± 0.016 | 323 ± 6 | 797 ± 29 | 323 ± 6 | 797 ± 29 | 12.3 ± 0.1 | 27.8 ± 2.4 | 2319 | 8.6 |
| 50 | 355 ± 3 | 0 | 0 | 5.263 ± 0.037 | 7267 ± 58 | 7833 ± 58 | 7267 ± 58 | 7833 ± 58 | 13.8 ± 0.1 | 36.0 ± 1.3 | 2319 | 8.5 |
| 100 | 405 ± 1 | 0 | 0 | 5.273 ± 0.009 | 17000 | 17000 | 17000 | 17000 | 14.0 ± 0.1 | 36.1 ± 1.3 | 2319 | 8.5 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 4.7 | 5.0 | 5.4 | 5.6 |
| fetch_chunks | 7.7 | 2.5 | 1.0 | 0.4 |
| fusion | 0.1 | 0.1 | 0.1 | 0.1 |
| queue_retrieval | 0.1 | 193.7 | 6781.8 | 14754.8 |
| rerank | 72.9 | 74.0 | 74.6 | 75.1 |
| search_bm25 | 36.0 | 36.4 | 36.3 | 36.6 |
| search_vector | 69.1 | 69.0 | 69.0 | 69.2 |
| server_e2e | 191.4 | 381.9 | 6969.3 | 14942.9 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 30s); server membatalkannya agar tidak membebani run berikutnya.
