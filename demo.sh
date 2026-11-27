#!/usr/bin/env bash
# Nyalakan seluruh demo dengan satu perintah: server LLM -> API -> UI, lalu buka http://127.0.0.1:7860
#   bash demo.sh
# Ctrl+C mematikan proses yang dinyalakan skrip ini (layanan yang sudah berjalan sebelumnya dibiarkan).
# Log: results/logs/{llm_server,server,ui}.log
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
LOG=results/logs
mkdir -p "$LOG"
started=()

cleanup() {
  if [ ${#started[@]} -gt 0 ]; then
    echo
    echo "Mematikan layanan demo yang dinyalakan skrip ini..."
    for p in "${started[@]}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

wait_http() {  # wait_http URL NAMA BATAS_DETIK
  for _ in $(seq 1 "${3:-180}"); do
    if curl -sf "$1" >/dev/null 2>&1; then echo "  ✓ $2 siap"; return 0; fi
    sleep 1
  done
  echo "  ✗ $2 tidak siap dalam ${3:-180} detik — lihat $LOG/" >&2
  return 1
}

if ! ls rag_hpc*.so >/dev/null 2>&1; then
  echo "⚠  Modul C++ rag_hpc belum dibangun: mode 'optimized' tidak tersedia. Jalankan dulu: bash cpp/build.sh"
fi

read -r backend llm_url api_url ui_url < <("$PY" -c "
from rag.config import load_config
c = load_config(); l = c['models']['llm']
print(l.get('backend'), l.get('base_url', ''), f\"http://{c['server']['host']}:{c['server']['port']}\",
      f\"http://{c['ui'].get('host', '127.0.0.1')}:{c['ui'].get('port', 7860)}\")")

echo "[1/3] Server LLM ($backend, $llm_url)"
if [ "$backend" = "openai" ]; then
  if curl -sf "$llm_url/health" >/dev/null 2>&1; then
    echo "  ✓ sudah berjalan"
  else
    "$PY" -m tools.llm_server --port "${llm_url##*:}" > "$LOG/llm_server.log" 2>&1 &
    started+=($!)
    wait_http "$llm_url/health" "llama-server" 180 || exit 1
  fi
elif [ "$backend" = "ollama" ]; then
  curl -sf "$llm_url/api/tags" >/dev/null 2>&1 && echo "  ✓ Ollama berjalan" || { echo "  ✗ jalankan dulu: ollama serve" >&2; exit 1; }
fi

echo "[2/3] API server ($api_url) — memuat model & warmup ±20 detik"
if curl -sf "$api_url/api/health" >/dev/null 2>&1; then
  echo "  ✓ sudah berjalan"
else
  "$PY" -W ignore -m server > "$LOG/server.log" 2>&1 &
  started+=($!)
  wait_http "$api_url/api/health" "API server" 300 || exit 1
fi

echo "[3/3] UI ($ui_url)"
if curl -sf "$ui_url" >/dev/null 2>&1; then
  echo "  ✓ sudah berjalan"
else
  "$PY" -W ignore -m ui > "$LOG/ui.log" 2>&1 &
  started+=($!)
  wait_http "$ui_url" "UI" 120 || exit 1
fi

echo
echo "Demo siap → $ui_url"
echo "  • Tab Chat: pilih mode baseline/optimized, ajukan pertanyaan, tekan \"Bandingkan mode\""
echo "  • Tab Monitor & Uji Beban: jalankan pengguna virtual sambil tetap chat"
if [ ${#started[@]} -gt 0 ]; then
  echo "Tekan Ctrl+C untuk mematikan."
  wait
fi
