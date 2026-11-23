#!/usr/bin/env bash
# Benchmark lengkap satu mode (metrik Tabel 3.2 proposal), berurutan agar tidak saling mengganggu:
#   1. uji beban retrieval saja — KB resmi UGM (koleksi ugm)
#   2. uji beban retrieval saja — 10^6 chunk sintetis (koleksi dummy_l)
#   3. waktu build indeks — dokumen resmi hasil scraping & 1.000 PDF sintetis
#   4. uji beban end-to-end dengan LLM — KB resmi UGM
#
# Prasyarat: `python -m server` sudah berjalan, koleksi ugm & dummy_l sudah diindeks.
# Server LLM dikelola otomatis: dimatikan pada tahap 1–3 (VRAM untuk retrieval/build) dan
# dinyalakan untuk tahap 4 (llama-server via `python -m tools.llm_server`, atau Ollama bila
# config memakai backend ollama).
#   bash loadtest/run_full.sh              # mode baseline
#   bash loadtest/run_full.sh optimized    # setelah modul C++ rag_hpc tersedia
#
# Variabel opsional: LEVELS="1 10 50 100", REPEATS=3, DUR_RETRIEVAL=60, DUR_LLM=120, STOP_LLM=60,
#   STEPS="1 2 3 4" (mis. STEPS=4 untuk mengulang tahap LLM saja)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-baseline}
PY=${PY:-.venv/bin/python}
LEVELS=${LEVELS:-"1 10 50 100"}
REPEATS=${REPEATS:-3}
DUR_RETRIEVAL=${DUR_RETRIEVAL:-60}
DUR_LLM=${DUR_LLM:-120}
STOP_LLM=${STOP_LLM:-60}
STEPS=${STEPS:-"1 2 3 4"}
PDF_DIR=data/dummy/files_pdf_1000

want() { [[ " $STEPS " == *" $1 "* ]]; }

warm() {  # warm KOLEKSI GENERATE(0/1) PERTANYAAN — muat snapshot indeks / model LLM sebelum diukur
  "$PY" - "$MODE" "$@" <<'EOF'
import sys, httpx
from rag.config import load_config
cfg = load_config()
mode, col, gen, q = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
url = f"http://{cfg['server']['host']}:{cfg['server']['port']}/api/chat"
r = httpx.post(url, json={"question": q, "collection": col, "mode": mode, "generate": gen,
                          "stream": False, "client": "warmup"}, timeout=900).json()
print(f"  pemanasan {col} (LLM={gen}): {r.get('e2e_ms', 0) / 1000:.1f}s", r.get("error") or "")
EOF
}

LLM_LOG=${LLM_LOG:-results/llm_server.log}

llm_info() {  # cetak "<backend> <base_url>" dari config
  "$PY" -c "from rag.config import load_config; l=load_config()['models']['llm']; print(l.get('backend'), l.get('base_url',''))"
}

unload_llm() {  # kosongkan VRAM dari model LLM selama tahap 1–3
  read -r backend url < <(llm_info)
  if [ "$backend" = "ollama" ]; then
    "$PY" - <<'EOF'
import httpx
from rag.config import load_config
llm = load_config()["models"]["llm"]
try:
    httpx.post(llm.get("base_url", "http://127.0.0.1:11434").rstrip("/") + "/api/generate",
               json={"model": llm["model"], "keep_alive": 0}, timeout=60)
    print("  model LLM (Ollama) dilepas dari VRAM")
except httpx.HTTPError as exc:
    print("  gagal melepas model LLM:", exc)
EOF
  else
    port=${url##*:}
    if pkill -f "llama-server.*--port ${port}" 2>/dev/null; then
      sleep 3
      echo "  llama-server (port ${port}) dihentikan untuk tahap retrieval/build"
    fi
  fi
}

ensure_llm() {  # pastikan server LLM siap sebelum tahap 4
  read -r backend url < <(llm_info)
  [ "$backend" = "ollama" ] && return 0
  if curl -sf "${url}/health" >/dev/null 2>&1; then return 0; fi
  unload_kb dummy_l  # bebaskan snapshot besar + cache VRAM PyTorch di server agar model LLM muat di GPU
  echo "  menyalakan llama-server: python -m tools.llm_server (log: $LLM_LOG)"
  nohup "$PY" -m tools.llm_server --port "${url##*:}" > "$LLM_LOG" 2>&1 &
  for _ in $(seq 1 180); do
    curl -sf "${url}/health" >/dev/null 2>&1 && { echo "  llama-server siap"; return 0; }
    sleep 1
  done
  echo "  llama-server tidak siap; lihat $LLM_LOG" >&2
  return 1
}

unload_kb() {  # lepas snapshot indeks koleksi dari RAM server (10^6 chunk ±2 GB)
  "$PY" - "$1" <<'EOF'
import sys, httpx
from rag.config import load_config
cfg = load_config()
r = httpx.post(f"http://{cfg['server']['host']}:{cfg['server']['port']}/api/kb/{sys.argv[1]}/unload", timeout=60).json()
print(f"  snapshot {r.get('unloaded')} dilepas: RSS server {r.get('rss_mb_before')} -> {r.get('rss_mb_after')} MB, "
      f"cache VRAM PyTorch {r.get('torch_reserved_mb')} MB")
EOF
}

if want 1 || want 2 || want 3; then
  unload_llm  # LLM tidak dipakai di tahap 1–3; VRAM yang tercatat murni milik retrieval
fi

if want 1; then
  echo "== [1/4] Uji beban retrieval saja — KB resmi (ugm) $(date +%T)"
  warm ugm 0 "$(grep -v '^#' loadtest/questions_ugm.txt | head -1)"
  "$PY" -m loadtest.run --mode "$MODE" --users $LEVELS --duration "$DUR_RETRIEVAL" --repeats "$REPEATS" \
    --no-generate --collection ugm
fi

if want 2; then
  echo "== [2/4] Uji beban retrieval saja — 10^6 chunk (dummy_l) $(date +%T)"
  warm dummy_l 0 "$(head -1 data/dummy/dummy_l/questions.txt)"
  "$PY" -m loadtest.run --mode "$MODE" --users $LEVELS --duration "$DUR_RETRIEVAL" --repeats "$REPEATS" \
    --no-generate --collection dummy_l --questions data/dummy/dummy_l/questions.txt
  unload_kb dummy_l
fi

if want 3; then
  echo "== [3/4] Waktu build indeks $(date +%T)"
  "$PY" -m tools.kb bench-build --mode "$MODE" --repeats "$REPEATS" data/scrape/raw
  [ -d "$PDF_DIR" ] || "$PY" -m tools.gen_dummy files --out "$PDF_DIR" --docs 1000 --format pdf
  "$PY" -m tools.kb bench-build --mode "$MODE" --repeats "$REPEATS" "$PDF_DIR"
fi

if want 4; then
  echo "== [4/4] Uji beban end-to-end dengan LLM — KB resmi (ugm) $(date +%T)"
  ensure_llm
  warm ugm 1 "$(grep -v '^#' loadtest/questions_ugm.txt | head -1)"
  "$PY" -m loadtest.run --mode "$MODE" --users $LEVELS --duration "$DUR_LLM" --repeats "$REPEATS" \
    --stop-timeout "$STOP_LLM" --collection ugm
fi

echo "== Selesai $(date +%T)"
