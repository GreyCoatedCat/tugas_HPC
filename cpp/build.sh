#!/usr/bin/env bash
# Bangun modul C++ rag_hpc (CUDA + OpenMP) -> rag_hpc.cpython-*.so di root proyek.
#   bash cpp/build.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
# nvcc 12.0 hanya mendukung host compiler GCC <= 12
HOST_CXX=${HOST_CXX:-$(command -v g++-12 || command -v g++)}

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$HOST_CXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
  -DPython_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  "$@"
cmake --build build -j
ls -la "$ROOT"/rag_hpc*.so
