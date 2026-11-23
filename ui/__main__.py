import argparse

from rag.config import load_config

from .app import CSS, build_ui

cfg = load_config()
ap = argparse.ArgumentParser(description="Jalankan demo UI Gradio")
ap.add_argument("--host", default=(cfg.get("ui") or {}).get("host", "127.0.0.1"))
ap.add_argument("--port", type=int, default=(cfg.get("ui") or {}).get("port", 7860))
args = ap.parse_args()

demo = build_ui()
demo.queue(default_concurrency_limit=32).launch(server_name=args.host, server_port=args.port, css=CSS, footer_links=[])
