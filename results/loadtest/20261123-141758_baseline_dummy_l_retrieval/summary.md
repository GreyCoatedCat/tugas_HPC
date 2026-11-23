# Hasil uji beban — 2026-11-23 14:32

- Mode: `baseline` | Koleksi: `dummy_l` | Generasi LLM: tidak (retrieval saja)
- Durasi per run: 60s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 ± 2 | 0 | 0 | 0.436 ± 0.026 | 257 ± 6 | 297 ± 15 | 257 ± 6 | 297 ± 15 | 3.2 ± 0.3 | 3.1 ± 1.0 | 2467 | 10.3 |
| 10 | 228 ± 2 | 0 | 0 | 3.751 ± 0.037 | 620 ± 44 | 1200 ± 100 | 620 ± 44 | 1200 ± 100 | 8.7 ± 0.1 | 21.0 ± 3.2 | 2463 | 10.2 |
| 50 | 278 ± 2 | 0 | 0 | 3.933 ± 0.016 | 10333 ± 577 | 11000 | 10333 ± 577 | 11000 | 9.1 ± 0.2 | 24.8 ± 3.5 | 2355 | 10.3 |
| 100 | 326 ± 2 | 0 | 0 ± 1 | 3.916 ± 0.032 | 23000 | 24000 | 23000 | 24000 | 9.3 ± 0.4 | 21.6 ± 4.8 | 2338 | 10.3 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 9.6 | 9.7 | 10.6 | 11.1 |
| fetch_chunks | 4.0 | 1.6 | 0.8 | 0.5 |
| fusion | 0.2 | 0.2 | 0.2 | 0.2 |
| queue_retrieval | 0.1 | 404.0 | 9542.2 | 19685.4 |
| rerank | 79.2 | 78.9 | 79.3 | 78.7 |
| search_bm25 | 52.2 | 51.6 | 52.0 | 52.0 |
| search_vector | 109.1 | 107.4 | 106.6 | 108.2 |
| server_e2e | 255.6 | 655.3 | 9793.6 | 19938.1 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 30s); server membatalkannya agar tidak membebani run berikutnya.
