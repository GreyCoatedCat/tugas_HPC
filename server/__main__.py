import ctypes
import sys

# glibc membuat arena malloc per thread (hingga 8 x jumlah core). Server melayani request dari
# ratusan thread, sehingga memori bebas tertahan di banyak arena dan RSS terus membengkak.
# Batasi jumlah arena sebelum thread apa pun dibuat (M_ARENA_MAX = -8).
if sys.platform.startswith("linux"):
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 4)
    except (OSError, AttributeError):
        pass

import argparse  # noqa: E402
import logging  # noqa: E402

import uvicorn  # noqa: E402

from rag.config import load_config  # noqa: E402

cfg = load_config()
ap = argparse.ArgumentParser(description="Jalankan API server RAG")
ap.add_argument("--host", default=cfg["server"]["host"])
ap.add_argument("--port", type=int, default=cfg["server"]["port"])
args = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info", access_log=False)
