"""Ukur kinerja server LLM (Ollama) untuk prompt RAG yang realistis.

Dipakai untuk membandingkan konfigurasi Ollama baseline vs teroptimasi
(OLLAMA_NUM_PARALLEL, OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE):

    python -m tools.llm_probe --concurrency 1 4 --label baseline-ollama

Konteks diambil dari API server (retrieval saja, generate=false) lalu dirakit dengan
prompt sistem yang sama seperti pipeline. Hasil: prefill/decode tok/s, throughput
agregat beberapa permintaan bersamaan, dan pembagian VRAM model (GPU vs CPU).
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

from rag.config import load_config
from rag.prompts import build_messages


def get_contexts(api: str, questions: list[str], collection: str) -> list[list[dict]]:
    out = []
    for q in questions:
        r = httpx.post(f"{api}/api/chat", json={"question": q, "collection": collection, "generate": False,
                                                 "stream": False, "client": "llm_probe"}, timeout=120).json()
        out.append([{**s, "n": s["n"]} for s in r.get("sources", [])])
    return out


def chat(base: str, payload: dict, openai: bool = False) -> dict:
    t = time.perf_counter()
    if not openai:
        r = httpx.post(f"{base}/api/chat", json=payload, timeout=900).json()
    else:  # llama-server / server OpenAI-compatible: samakan nama field dengan Ollama
        o = httpx.post(f"{base}/v1/chat/completions", json=payload, timeout=900).json()
        tm = o.get("timings") or {}
        r = {"eval_count": tm.get("predicted_n") or (o.get("usage") or {}).get("completion_tokens", 0),
             "eval_duration": (tm.get("predicted_ms") or 0) * 1e6,
             "prompt_eval_count": tm.get("prompt_n") or (o.get("usage") or {}).get("prompt_tokens", 0),
             "prompt_eval_duration": (tm.get("prompt_ms") or 0) * 1e6}
    r["_wall_ms"] = (time.perf_counter() - t) * 1000
    return r


def main(argv=None) -> None:
    cfg = load_config()
    llm = cfg["models"]["llm"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--label", default="probe")
    ap.add_argument("--collection", default="ugm")
    ap.add_argument("--api", default=f"http://{cfg['server']['host']}:{cfg['server']['port']}")
    ap.add_argument("--openai", metavar="URL", help="ukur server OpenAI-compatible (mis. llama-server) alih-alih Ollama")
    a = ap.parse_args(argv)

    base = (a.openai or llm.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
    is_oai = bool(a.openai)
    qs = [l.strip() for l in Path("loadtest/questions_ugm.txt").read_text(encoding="utf-8").splitlines()
          if l.strip() and not l.startswith("#")]
    nmax = max(a.concurrency) * a.rounds
    ctxs = get_contexts(a.api, qs[:nmax], a.collection)
    refuse = (cfg.get("grounding") or {}).get("refuse_message", "")
    options = {"temperature": llm.get("temperature", 0.1), "num_predict": llm.get("max_tokens", 768)}
    if llm.get("num_ctx"):
        options["num_ctx"] = int(llm["num_ctx"])

    def payload(i: int) -> dict:
        msgs = build_messages(qs[i], ctxs[i], refuse)
        if is_oai:
            return {"model": llm["model"], "messages": msgs, "stream": False, "temperature": options["temperature"],
                    "max_tokens": options["num_predict"]}
        p = {"model": llm["model"], "messages": msgs, "stream": False,
             "options": options, "keep_alive": llm.get("keep_alive", "30m")}
        if llm.get("think") is not None:
            p["think"] = bool(llm["think"])
        return p

    chat(base, payload(0), is_oai)  # muat model + pemanasan
    vram: dict = {}
    if not is_oai:
        ps = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [{}])[0]
        vram = {"size_mb": round(ps.get("size", 0) / 2**20), "size_vram_mb": round(ps.get("size_vram", 0) / 2**20)}
        vram["gpu_fraction"] = round(vram["size_vram_mb"] / vram["size_mb"], 3) if vram["size_mb"] else None
        print(f"[{a.label}] model {llm['model']}: {vram['size_vram_mb']}/{vram['size_mb']} MB di GPU ({vram['gpu_fraction']:.0%})")

    report = {"label": a.label, "vram": vram, "runs": {}}
    for c in a.concurrency:
        rows, walls = [], []
        for rnd in range(a.rounds):
            res: list[dict] = [None] * c  # type: ignore[list-item]
            idx = [rnd * c + j for j in range(c)]
            ths = [threading.Thread(target=lambda j=j, i=i: res.__setitem__(j, chat(base, payload(i), is_oai))) for j, i in enumerate(idx)]
            t = time.perf_counter()
            [th.start() for th in ths]
            [th.join() for th in ths]
            walls.append((time.perf_counter() - t) * 1000)
            rows.extend(res)
        dec = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in rows if r.get("eval_duration")]
        pre = [r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9) for r in rows if r.get("prompt_eval_duration")]
        toks = sum(r.get("eval_count", 0) for r in rows)
        out = {
            "requests": len(rows),
            "decode_tok_s_per_request": round(statistics.fmean(dec), 1),
            "prefill_tok_s_per_request": round(statistics.fmean(pre), 1),
            "aggregate_decode_tok_s": round(toks / (sum(walls) / 1000), 1),
            "wall_ms_per_round": round(statistics.fmean(walls)),
            "e2e_ms_mean": round(statistics.fmean(r["_wall_ms"] for r in rows)),
            "completion_tokens_mean": round(statistics.fmean(r.get("eval_count", 0) for r in rows)),
            "prompt_tokens_mean": round(statistics.fmean(r.get("prompt_eval_count", 0) for r in rows)),
        }
        report["runs"][str(c)] = out
        print(f"[{a.label}] konkurensi {c}: decode {out['decode_tok_s_per_request']} tok/s/req, "
              f"agregat {out['aggregate_decode_tok_s']} tok/s, prefill {out['prefill_tok_s_per_request']} tok/s, "
              f"e2e rata2 {out['e2e_ms_mean'] / 1000:.1f}s, ronde {out['wall_ms_per_round'] / 1000:.1f}s")
    out_dir = Path(cfg["paths"]["results_dir"]) / "llm_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a.label}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Hasil: {path}")


if __name__ == "__main__":
    main()
