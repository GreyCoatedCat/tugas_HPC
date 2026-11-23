#!/usr/bin/env bash
# Evaluasi sebelum vs sesudah optimasi (±2,5 jam), berurutan agar tidak saling mengganggu:
#   1. ablation per optimasi (tools.ablation, di dalam proses, tanpa LLM)
#   2. run_full.sh tahap 1–3 untuk baseline lalu optimized (retrieval ugm & 10^6, build indeks)
#   3. run_full.sh tahap 4 (end-to-end dengan LLM) untuk baseline lalu optimized
#   4. laporan perbandingan -> results/comparison.md
# Prasyarat: `python -m server` berjalan. Tutup aplikasi berat (browser) agar RAM cukup.
#   bash loadtest/run_compare.sh
#   SKIP="A B" bash loadtest/run_compare.sh     # lanjutkan: lewati tahap yang sudah selesai hari ini
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SINCE=$(date +%Y%m%d)
# folder hasil end-to-end baseline asli (Ollama bawaan, 20 Nov 2026)
LLM_REF=${LLM_REF:-results/loadtest/20261120-120322_baseline_ugm}
[ -f "$LLM_REF/summary.json" ] || LLM_REF=""

SKIP=${SKIP:-}
run() { [[ " $SKIP " == *" $1 "* ]] && { echo "=== [$1] dilewati"; return 1; }; return 0; }

if run A; then
  echo "=== [A] Ablation $(date +%T)"
  pkill -f "llama-server.*--port" 2>/dev/null && sleep 3 || true  # VRAM untuk ablation (LLM tidak dipakai)
  "$PY" -m tools.ablation --collections ugm dummy_l
fi
if run B; then
  echo "=== [B] Retrieval & build: baseline $(date +%T)"
  STEPS="1 2 3" bash loadtest/run_full.sh baseline
fi
if run C; then
  echo "=== [C] Retrieval & build: optimized $(date +%T)"
  STEPS="1 2 3" bash loadtest/run_full.sh optimized
fi
if run D; then
  echo "=== [D] End-to-end LLM: baseline $(date +%T)"
  STEPS="4" bash loadtest/run_full.sh baseline
fi
if run E; then
  echo "=== [E] End-to-end LLM: optimized $(date +%T)"
  STEPS="4" bash loadtest/run_full.sh optimized
fi

echo "=== [F] Laporan $(date +%T)"
"$PY" -m tools.compare --since "$SINCE" ${LLM_REF:+--llm-ref "$LLM_REF"}
echo "=== Selesai $(date +%T)"
