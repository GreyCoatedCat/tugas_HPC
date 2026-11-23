# Hasil uji beban — 2026-11-23 14:17

- Mode: `baseline` | Koleksi: `ugm` | Generasi LLM: tidak (retrieval saja)
- Durasi per run: 60s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 ± 1 | 0 | 0 | 0.476 ± 0.019 | 140 | 153 ± 6 | 140 | 153 ± 6 | 2.9 ± 0.5 | 3.7 ± 1.0 | 2217 | 8.2 |
| 10 | 266 ± 4 | 0 | 0 | 4.397 ± 0.063 | 197 ± 6 | 530 ± 26 | 197 ± 6 | 530 ± 26 | 7.1 ± 0.6 | 50.9 ± 3.5 | 2240 | 8.1 |
| 50 | 417 ± 2 | 0 | 0 | 6.330 ± 0.008 | 5733 ± 58 | 6200 | 5733 ± 58 | 6200 | 10.5 ± 0.2 | 79.6 ± 0.9 | 2253 | 8.1 |
| 100 | 463 ± 3 | 0 | 1 ± 1 | 6.283 ± 0.026 | 14000 | 14000 | 14000 | 14000 | 10.9 ± 0.1 | 78.5 ± 1.1 | 2400 | 8.2 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 8.6 | 7.1 | 9.8 | 10.3 |
| fetch_chunks | 0.3 | 0.2 | 0.3 | 0.3 |
| fusion | 0.1 | 0.1 | 0.1 | 0.1 |
| queue_retrieval | 0.1 | 106.3 | 5376.2 | 12229.3 |
| rerank | 123.8 | 134.8 | 143.6 | 144.1 |
| search_bm25 | 0.1 | 0.1 | 0.2 | 0.2 |
| search_vector | 0.4 | 0.4 | 0.5 | 0.5 |
| server_e2e | 134.4 | 250.3 | 5532.8 | 12387.1 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 30s); server membatalkannya agar tidak membebani run berikutnya.
