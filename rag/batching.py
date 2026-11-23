"""Dynamic batching untuk tahap retrieval (mode teroptimasi).

Permintaan konkuren masuk ke antrean FIFO. Satu thread eksekutor mengambil SEMUA permintaan
yang sedang menunggu (maks ``retrieval_batch_max``, koleksi yang sama) begitu eksekutor
bebas, lalu memprosesnya sebagai satu batch:

- embedding semua kueri dalam satu forward pass (FP16);
- pencarian vektor semua kueri dalam satu peluncuran kernel CUDA (matriks dibaca sekali);
- BM25 (CPU, OpenMP) berjalan BERSAMAAN dengan pencarian vektor (GPU) — overlap CPU–GPU;
- rerank semua pasangan (kueri, kandidat) dalam batch besar FP16.

Tidak ada jendela tunggu buatan secara default (``retrieval_batch_window_ms: 0``): pada beban
rendah permintaan langsung diproses sendiri, sedangkan pada beban tinggi batch terbentuk dengan
sendirinya dari permintaan yang menumpuk selama batch sebelumnya berjalan (continuous batching).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import ModeConfig
from .timing import StageTimer

if TYPE_CHECKING:  # pragma: no cover
    from .service import RAGService

log = logging.getLogger(__name__)


@dataclass(eq=False)
class _Req:
    question: str
    collection: str
    timer: StageTimer
    cancel: threading.Event | None
    t_enqueue: float
    fut: Future = field(default_factory=Future)


class BatchRetriever:
    def __init__(self, svc: "RAGService", mode: ModeConfig) -> None:
        from .service import RequestCancelled  # hindari impor melingkar

        self._cancelled = RequestCancelled
        self.svc = svc
        self.mode = mode
        self.max_batch = max(1, int(mode.retrieval_batch_max))
        self.window_s = max(0.0, float(mode.retrieval_batch_window_ms)) / 1000.0
        self._q: deque[_Req] = deque()
        self._cv = threading.Condition()
        self._stop = False
        self.stats: dict[str, Any] = {"batches": 0, "requests": 0, "max_batch_seen": 0}
        self._thread = threading.Thread(target=self._loop, name=f"batcher-{mode.name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ API
    def run(self, question: str, collection: str, timer: StageTimer, cancel: threading.Event | None = None):
        req = _Req(question, collection, timer, cancel, time.perf_counter())
        with self._cv:
            self._q.append(req)
            self._cv.notify()
        while True:
            try:
                return req.fut.result(timeout=0.2)
            except FutureTimeout:
                if cancel is not None and cancel.is_set():
                    with self._cv:
                        if req in self._q:  # belum diambil eksekutor -> keluarkan dari antrean
                            self._q.remove(req)
                            raise self._cancelled() from None
                    # sudah masuk batch yang sedang berjalan: tunggu sebentar sampai selesai

    def queued(self) -> int:
        return len(self._q)

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # ------------------------------------------------------------ eksekutor
    def _take_batch(self) -> list[_Req]:
        with self._cv:
            while not self._q and not self._stop:
                self._cv.wait()
            if self._stop:
                return []
            first = self._q.popleft()
            batch = [first]
            deadline = time.perf_counter() + self.window_s
            while len(batch) < self.max_batch:
                for r in [r for r in self._q if r.collection == first.collection][: self.max_batch - len(batch)]:
                    self._q.remove(r)
                    batch.append(r)
                remaining = deadline - time.perf_counter()
                if len(batch) >= self.max_batch or remaining <= 0:
                    break
                self._cv.wait(remaining)
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                return
            live = []
            for r in batch:
                if r.cancel is not None and r.cancel.is_set():
                    r.fut.set_exception(self._cancelled())
                else:
                    live.append(r)
            if not live:
                continue
            n_live = len(live)
            t_start = time.perf_counter()
            for r in live:
                r.timer.add("queue_retrieval", (t_start - r.t_enqueue) * 1000)
            try:
                results, snap, info = self.svc.retrieve_batch(
                    [r.question for r in live], live[0].collection, self.mode, [r.timer for r in live]
                )
                for r, hits in zip(live, results):
                    r.fut.set_result((hits, snap, info))
                # jangan tahan snapshot di frame thread ini: snapshot yang di-unload harus bisa
                # dibebaskan (RAM + matriks di VRAM) walau tidak ada batch berikutnya
                del results, snap, info
            except Exception as exc:  # noqa: BLE001
                log.exception("batch retrieval gagal")
                for r in live:
                    if not r.fut.done():
                        r.fut.set_exception(exc)
            # `r` terakhir masih menunjuk Future yang hasilnya memuat snapshot -> lepas juga
            batch = live = r = hits = None  # type: ignore[assignment]
            self.stats["batches"] += 1
            self.stats["requests"] += n_live
            self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], n_live)
