# Hasil uji beban — 2026-11-23 15:04

- Mode: `optimized` | Koleksi: `dummy_l` | Generasi LLM: tidak (retrieval saja)
- Durasi per run: 60s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 ± 2 | 0 | 0 | 0.454 ± 0.025 | 60 ± 3 | 70 ± 4 | 60 ± 3 | 70 ± 4 | 11.5 ± 0.6 | 15.6 ± 1.6 | 4394 | 10.3 |
| 10 | 286 ± 2 | 0 | 0 | 4.747 ± 0.033 | 61 ± 2 | 147 ± 46 | 61 ± 2 | 147 ± 46 | 13.3 ± 1.9 | 26.9 ± 3.3 | 4849 | 10.6 |
| 50 | 1269 ± 15 | 0 | 1 ± 1 | 20.994 ± 0.273 | 343 ± 6 | 697 ± 12 | 343 ± 6 | 697 ± 12 | 20.0 ± 3.5 | 78.6 ± 2.0 | 5107 | 10.7 |
| 100 | 1640 ± 8 | 0 | 0 | 26.587 ± 0.160 | 1733 ± 58 | 2100 | 1733 ± 58 | 2100 | 19.8 ± 3.3 | 75.9 ± 0.7 | 4873 | 10.6 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 12.1 | 10.2 | 10.4 | 15.7 |
| fetch_chunks | 0.4 | 0.4 | 1.1 | 2.7 |
| fusion | 0.2 | 0.2 | 0.5 | 1.1 |
| queue_retrieval | 0.1 | 8.4 | 127.4 | 1133.5 |
| rerank | 35.3 | 44.8 | 210.0 | 510.0 |
| search_bm25 | 5.9 | 5.9 | 19.5 | 60.0 |
| search_vector | 5.3 | 5.7 | 11.5 | 16.9 |
| server_e2e | 55.5 | 72.1 | 373.2 | 1731.4 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 30s); server membatalkannya agar tidak membebani run berikutnya.
