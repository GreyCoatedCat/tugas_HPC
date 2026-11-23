"""Jalankan server LLM teroptimasi: llama-server (llama.cpp bawaan Ollama) + continuous batching.

Temuan profiling pada Ollama bawaan (lihat README, bagian optimasi LLM):
- Ollama menolak permintaan paralel untuk arsitektur ``qwen35`` ("model architecture does not
  currently support parallel requests"), sehingga semua permintaan dilayani satu per satu.
- Ollama menyisihkan ±1,3 GB VRAM untuk vision projector (mmproj) yang tidak kita pakai,
  sehingga hanya 25/34 layer muat di GPU dan sisanya dihitung di CPU.

Skrip ini menjalankan ``llama-server`` yang sama dengan yang dipakai Ollama di balik layar,
dengan berkas GGUF yang sama (bobot identik), tetapi:
- ``--parallel N --cont-batching``: N slot, token dari beberapa permintaan di-decode dalam satu batch;
- ``--no-mmproj`` + ``--fit``: tanpa vision projector; layer sebanyak mungkin di GPU (semua 34 layer
  bila VRAM bebas cukup, sisanya di CPU bila aplikasi desktop memakai VRAM);
- ``--flash-attn on -ctk q8_0 -ctv q8_0``: flash attention + KV cache 8-bit (hemat VRAM);
- ``--cache-ram 0``: tanpa cache prompt di RAM host (default llama.cpp 8 GB) — hemat RAM dan
  pengukuran tidak terpengaruh pertanyaan yang pernah diajukan sebelumnya.

API-nya kompatibel OpenAI (``/v1/chat/completions``) -> backend ``openai`` di config.yaml.

    python -m tools.llm_server              # 4 slot, port 8081
    python -m tools.llm_server --parallel 1 # satu slot (untuk ablation)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx


def find_blob(model: str, models_dir: Path) -> Path:
    name, _, tag = model.partition(":")
    ns = "library" if "/" not in name else ""
    manifest = models_dir / "manifests" / "registry.ollama.ai" / (ns or "") / name / (tag or "latest")
    if not manifest.exists():
        raise SystemExit(f"Manifest {manifest} tidak ada. Unduh dulu: ollama pull {model}")
    layers = json.loads(manifest.read_text())["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"] == "application/vnd.ollama.image.model")
    blob = models_dir / "blobs" / digest.replace(":", "-")
    if not blob.exists():
        raise SystemExit(f"Berkas model {blob} tidak ditemukan")
    return blob


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--parallel", type=int, default=4, help="jumlah slot (= llm_max_concurrency mode optimized)")
    ap.add_argument("--ctx-per-slot", type=int, default=4096)
    ap.add_argument("--gpu-layers", default="auto",
                    help="'auto' = llama.cpp memilih layer sebanyak yang muat di VRAM bebas (margin --fit-target); "
                         "angka = paksa (mis. 99 = semua layer)")
    ap.add_argument("--fit-target", type=int, default=256, help="sisa VRAM minimal (MiB) untuk mode auto")
    ap.add_argument("--cache-ram", type=int, default=0,
                    help="cache prompt di RAM host (MiB). Default llama.cpp 8192 MiB: memakan RAM besar dan membuat "
                         "hasil benchmark bergantung pada pertanyaan sebelumnya -> 0 = nonaktif")
    ap.add_argument("--ollama-lib", default=os.environ.get("OLLAMA_LIB", "/usr/local/lib/ollama"))
    ap.add_argument("--models-dir", default=os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    lib = Path(a.ollama_lib)
    server = lib / "llama-server"
    if not server.exists():
        raise SystemExit(f"{server} tidak ditemukan (butuh Ollama >= 0.30 yang membawa llama-server)")
    cuda_dir = next((lib / d for d in ("cuda_v13", "cuda_v12") if (lib / d / "libggml-cuda.so").exists()), None)
    if cuda_dir is None:
        raise SystemExit(f"Backend CUDA ggml tidak ditemukan di {lib}")
    blob = find_blob(a.model, Path(a.models_dir))

    # Lepas model dari Ollama (bila masih berjalan) agar VRAM tidak terpakai dua kali.
    try:
        httpx.post("http://127.0.0.1:11434/api/generate", json={"model": a.model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass

    cmd = [
        str(server), "--model", str(blob), "--alias", a.model, "--host", "127.0.0.1", "--port", str(a.port),
        "--no-webui", "-c", str(a.parallel * a.ctx_per_slot), "-np", str(a.parallel), "-cb",
        "--jinja", "--reasoning", "off", "--no-mmproj", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
        "-b", "512", "-ub", "512", "--metrics", "--cache-ram", str(a.cache_ram),
    ]
    if a.gpu_layers == "auto":
        # Aplikasi desktop (browser, terminal, Xorg) ikut memakai VRAM; jumlah layer di GPU disesuaikan
        # dengan memori yang benar-benar bebas agar server tidak gagal start.
        cmd += ["--fit", "on", "--fit-target", str(a.fit_target)]
    else:
        cmd += ["-ngl", str(a.gpu_layers)]
    env = {**os.environ, "LD_LIBRARY_PATH": f"{lib}:{cuda_dir}", "GGML_BACKEND_PATH": str(cuda_dir / "libggml-cuda.so")}
    print("Menjalankan:", " ".join(cmd), flush=True)
    if a.dry_run:
        return
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], check=False)
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
