# Hasil uji beban — 2026-11-20 12:37

- Mode: `baseline` | Koleksi: `ugm` | Generasi LLM: ya
- Durasi per run: 120s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 ± 2 | 0 | 0 | 0.045 ± 0.014 | 15833 ± 8129 | 30667 ± 2517 | 1427 ± 1363 | 2933 ± 503 | 40.3 ± 1.2 | 25.9 ± 1.1 | 7432 | 10.5 |
| 10 | 7 ± 2 | 0 | 8 ± 1 | 0.041 ± 0.008 | 91333 ± 16258 | 155000 ± 19975 | 72000 ± 3606 | 140667 ± 11719 | 44.2 ± 0.1 | 27.4 ± 1.6 | 7432 | 11.9 |
| 50 | 7 ± 1 | 0 | 48 ± 1 | 0.037 ± 0.006 | 101333 ± 6429 | 166333 ± 11240 | 72667 ± 20306 | 145000 ± 19975 | 46.6 ± 3.0 | 28.2 ± 1.7 | 7560 | 12.5 |
| 100 | 7 ± 2 | 0 | 97 ± 1 | 0.041 ± 0.008 | 98667 ± 13577 | 165333 ± 9504 | 85667 ± 18037 | 156000 ± 10536 | 46.3 ± 0.4 | 32.5 ± 1.2 | 7696 | 10.9 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 6.8 | 8.3 | 12.3 | 15.2 |
| fetch_chunks | 0.2 | 0.2 | 0.2 | 0.2 |
| fusion | 0.1 | 0.1 | 0.1 | 0.1 |
| llm_decode | 14819.0 | 19738.8 | 24603.0 | 22378.3 |
| llm_prefill | 1595.8 | 1603.2 | 633.8 | 604.1 |
| queue_llm | 0.0 | 65017.3 | 67666.6 | 79296.3 |
| queue_retrieval | 0.1 | 465.0 | 428.9 | 516.9 |
| rerank | 120.5 | 169.5 | 167.7 | 167.4 |
| search_bm25 | 0.1 | 0.1 | 0.1 | 0.2 |
| search_vector | 0.3 | 0.5 | 0.5 | 0.5 |
| server_e2e | 16544.0 | 87007.1 | 93516.5 | 102981.6 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 60s); server membatalkannya agar tidak membebani run berikutnya.
