# Hasil uji beban — 2026-11-23 14:51

- Mode: `optimized` | Koleksi: `ugm` | Generasi LLM: tidak (retrieval saja)
- Durasi per run: 60s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 ± 1 | 0 | 0 ± 1 | 0.491 ± 0.019 | 79 ± 2 | 88 ± 1 | 79 ± 2 | 88 ± 1 | 3.1 ± 0.2 | 3.2 ± 2.4 | 2288 | 8.3 |
| 10 | 281 ± 1 | 0 | 0 | 4.655 ± 0.020 | 82 ± 4 | 263 ± 35 | 82 ± 4 | 263 ± 35 | 7.3 ± 0.1 | 19.3 ± 4.7 | 3049 | 8.2 |
| 50 | 852 ± 33 | 0 | 0 | 13.944 ± 0.599 | 1533 ± 153 | 2267 ± 208 | 1533 ± 153 | 2267 ± 208 | 13.2 ± 2.6 | 85.8 ± 1.6 | 3242 | 8.7 |
| 100 | 880 ± 27 | 0 | 1 ± 1 | 13.557 ± 0.395 | 5233 ± 231 | 5967 ± 306 | 5233 ± 231 | 5967 ± 306 | 13.1 ± 0.7 | 85.0 ± 0.6 | 3815 | 9.2 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 11.9 | 12.2 | 14.6 | 15.8 |
| fetch_chunks | 0.3 | 0.4 | 0.9 | 0.9 |
| fusion | 0.2 | 0.2 | 0.9 | 1.0 |
| queue_retrieval | 0.2 | 17.9 | 557.0 | 3939.8 |
| rerank | 59.3 | 76.8 | 971.9 | 1130.2 |
| search_bm25 | 0.3 | 0.3 | 0.9 | 0.8 |
| search_vector | 0.4 | 0.3 | 0.3 | 0.3 |
| server_e2e | 74.1 | 109.8 | 1553.7 | 5097.5 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 30s); server membatalkannya agar tidak membebani run berikutnya.
