# Hasil uji beban — 2026-11-23 16:40

- Mode: `baseline` | Koleksi: `ugm` | Generasi LLM: ya
- Durasi per run: 120s | Pengulangan: 3 | Jeda pengguna: 1.0–3.0s | Host: http://127.0.0.1:8000

| Pengguna | Selesai | Gagal | Ditinggal | Throughput (QPS) | E2E p50 (ms) | E2E p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | CPU % | GPU % | VRAM maks (MB) | RAM maks (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 ± 1 | 0 | 0 | 0.027 ± 0.003 | 27333 ± 1528 | 40333 ± 4619 | 1533 ± 115 | 1767 ± 115 | 21.2 ± 0.3 | 91.6 ± 0.9 | 7748 | 7.4 |
| 10 | 7 ± 1 | 0 | 7 ± 1 | 0.041 ± 0.006 | 112667 ± 12097 | 167333 ± 15144 | 90000 ± 7000 | 140333 ± 12097 | 24.3 ± 2.8 | 98.2 ± 0.3 | 7867 | 8.8 |
| 50 | 8 | 0 | 47 ± 1 | 0.044 | 98333 ± 8145 | 169333 ± 4041 | 73667 ± 9074 | 153667 ± 8737 | 25.3 ± 0.8 | 98.3 ± 0.1 | 8097 | 10.1 |
| 100 | 7 ± 2 | 0 | 97 ± 1 | 0.037 ± 0.008 | 105333 ± 18148 | 169000 ± 2646 | 87333 ± 26026 | 144333 ± 4933 | 24.5 ± 0.4 | 98.3 ± 0.3 | 7979 | 9.5 |

## Latensi rata-rata per tahap (ms, sisi server)

| Tahap | 1 pengguna | 10 pengguna | 50 pengguna | 100 pengguna |
|---|---:|---:|---:|---:|
| embed_query | 10.3 | 7.2 | 11.4 | 13.2 |
| fetch_chunks | 0.3 | 0.2 | 0.2 | 0.2 |
| fusion | 0.2 | 0.1 | 0.1 | 0.1 |
| llm_decode | 25227.4 | 21438.6 | 19551.7 | 24177.8 |
| llm_prefill | 1333.7 | 1540.1 | 1601.8 | 2179.6 |
| queue_llm | 0.0 | 76630.4 | 70875.2 | 72539.1 |
| queue_retrieval | 0.1 | 793.6 | 980.5 | 723.8 |
| rerank | 118.8 | 272.5 | 302.6 | 278.4 |
| search_bm25 | 0.2 | 0.1 | 0.1 | 0.1 |
| search_vector | 0.5 | 0.3 | 0.4 | 0.4 |
| server_e2e | 26692.8 | 100684.7 | 93325.6 | 99914.4 |

Catatan: persentil dari Locust dibulatkan ke 2 digit signifikan. Throughput = permintaan selesai / waktu run sebenarnya (durasi + masa tunggu akhir). Latensi & latensi per tahap hanya dari request yang selesai. *Ditinggal* = request yang masih antre/berjalan saat run dihentikan (setelah stop-timeout 60s); server membatalkannya agar tidak membebani run berikutnya.
