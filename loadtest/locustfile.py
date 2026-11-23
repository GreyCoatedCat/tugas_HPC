"""Skenario uji beban Locust untuk API chatbot.

Setiap pengguna virtual mengirim pertanyaan acak ke POST /api/chat (stream
NDJSON) lalu mencatat metrik kustom:

    METRIC/e2e         latensi end-to-end sisi klien (kirim -> stream selesai)
    METRIC/ttft        time-to-first-token sisi klien
    STAGE/<tahap>      latensi per tahap pipeline yang dilaporkan server
    STAGE/server_e2e   latensi end-to-end sisi server
    METRIC/abandoned   request yang masih antre/berjalan saat run dihentikan (stop-timeout
                       habis) — tidak dihitung sebagai latensi, hanya jumlahnya

Catatan: entri "POST /api/chat" bawaan Locust hanya mengukur waktu sampai
header respons diterima (karena stream=True) — gunakan METRIC/e2e.

Konfigurasi lewat environment variable:
    RAG_QUESTIONS   file pertanyaan (1 per baris)       default: questions_ugm.txt
    RAG_COLLECTION  koleksi KB                           default: koleksi default server
    RAG_MODE        baseline | optimized                 default: baseline
    RAG_GENERATE    1 = dengan LLM, 0 = retrieval saja   default: 1
    RAG_WAIT_MIN / RAG_WAIT_MAX  jeda antar pertanyaan per pengguna (detik), default 1..3

Jalankan langsung (UI web Locust di http://localhost:8089):
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from gevent import GreenletExit
from locust import HttpUser, between, task

QUESTIONS_FILE = os.environ.get("RAG_QUESTIONS") or str(Path(__file__).with_name("questions_ugm.txt"))
COLLECTION = os.environ.get("RAG_COLLECTION") or None
MODE = os.environ.get("RAG_MODE", "baseline")
GENERATE = os.environ.get("RAG_GENERATE", "1") != "0"
WAIT_MIN = float(os.environ.get("RAG_WAIT_MIN", "1"))
WAIT_MAX = float(os.environ.get("RAG_WAIT_MAX", "3"))
# CSV Locust ditulis sebelum pengguna dibunuh, jadi jumlah request yang ditinggal juga
# ditulis ke file terpisah (dipakai loadtest/run.py).
ABANDONED_FILE = os.environ.get("RAG_ABANDONED_FILE")
_abandoned = 0

QUESTIONS = [
    ln.strip()
    for ln in Path(QUESTIONS_FILE).read_text(encoding="utf-8").splitlines()
    if ln.strip() and not ln.lstrip().startswith("#")
]


class RagUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)

    def _metric(self, rtype: str, name: str, ms: float | None, exc: Exception | None = None) -> None:
        if ms is None:
            return
        self.environment.events.request.fire(
            request_type=rtype, name=name, response_time=ms, response_length=0, exception=exc, context={}
        )

    @task
    def ask(self) -> None:
        t0 = time.perf_counter()
        try:
            self._ask(t0)
        except GreenletExit:
            # Locust membunuh pengguna saat stop-timeout habis: catat sebagai request yang ditinggal.
            global _abandoned
            _abandoned += 1
            self._metric("METRIC", "abandoned", (time.perf_counter() - t0) * 1000)
            if ABANDONED_FILE:
                Path(ABANDONED_FILE).write_text(str(_abandoned), encoding="utf-8")
            raise

    def _ask(self, t0: float) -> None:
        payload = {
            "question": random.choice(QUESTIONS),
            "collection": COLLECTION,
            "mode": MODE,
            "generate": GENERATE,
            "stream": True,
            "client": "locust",
        }
        ttft = None
        done = None
        error = None
        with self.client.post(
            "/api/chat", json=payload, stream=True, catch_response=True, name="/api/chat", timeout=1800
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    ev = json.loads(line)
                    kind = ev.get("type")
                    if kind == "token" and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    elif kind == "done":
                        done = ev
                    elif kind == "error":
                        error = ev.get("message", "error")
            except Exception as exc:  # noqa: BLE001
                error = f"stream terputus: {exc}"
            if error or done is None:
                resp.failure(error or "stream berakhir tanpa event 'done'")
            else:
                resp.success()
        e2e = (time.perf_counter() - t0) * 1000
        if error or done is None:
            self._metric("METRIC", "e2e", e2e, RuntimeError(error or "incomplete"))
            return
        self._metric("METRIC", "e2e", e2e)
        self._metric("METRIC", "ttft", ttft if ttft is not None else e2e)
        for stage, ms in (done.get("timings_ms") or {}).items():
            self._metric("STAGE", stage, ms)
        self._metric("STAGE", "server_e2e", done.get("e2e_ms"))
