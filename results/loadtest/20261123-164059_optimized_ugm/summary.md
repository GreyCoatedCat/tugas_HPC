# Hasil uji beban — 2026-11-23 17:15

- Mode: `optimized` | Koleksi: `ugm` | Generasi LLM: ya
- Durasi per run: 120s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 ± 1 | 0 | 0 | 0.039 ± 0.006 | 17333 ± 5033 | 36333 ± 9074 | 1833 ± 115 | 2233 ± 513 | 21.4 ± 0.3 | 90.4 ± 2.3 | 7773 | 7.9 |
| 10 | 14 ± 3 | 0 | 5 ± 1 | 0.080 ± 0.014 | 92333 ± 18448 | 134667 ± 17926 | 53667 ± 6429 | 84000 ± 19975 | 22.1 ± 1.2 | 96.2 ± 0.6 | 8117 | 8.6 |
| 50 | 13 ± 1 | 0 | 44 ± 2 | 0.074 ± 0.006 | 116000 ± 18083 | 171667 ± 8737 | 58333 ± 15044 | 137000 ± 29547 | 21.9 ± 0.3 | 96.8 ± 0.6 | 8088 | 8.5 |
| 100 | 15 ± 2 | 0 | 95 ± 1 | 0.081 ± 0.008 | 100667 ± 5686 | 171667 ± 5508 | 61333 ± 17926 | 140667 ± 15535 | 21.8 ± 0.5 | 97.2 ± 0.4 | 8085 | 8.0 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 10.5 | 11.6 | 30.0 | 14.1 |
| fetch_chunks | 0.2 | 0.7 | 1.0 | 0.8 |
| fusion | 0.1 | 0.4 | 1.8 | 1.1 |
| llm_decode | 18071.8 | 39450.3 | 41801.1 | 35522.1 |
| llm_prefill | 1422.2 | 2749.5 | 3654.5 | 4140.1 |
| queue_llm | 0.0 | 40756.8 | 57460.4 | 56094.8 |
| queue_retrieval | 0.1 | 74.6 | 216.9 | 186.5 |
| rerank | 49.4 | 486.8 | 1517.6 | 1651.4 |
| search_bm25 | 0.3 | 0.7 | 1.0 | 1.1 |
| search_vector | 0.2 | 0.9 | 1.6 | 1.1 |
| server_e2e | 19556.1 | 83535.9 | 104692.4 | 97619.6 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 60s); server membatalkannya agar tidak membebani run berikutnya.
