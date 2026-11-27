"""Antrean slot pipeline/LLM harus FIFO dan bisa dibatalkan tanpa membocorkan slot."""

import threading
import time

import pytest

from rag.service import FifoSemaphore, RequestCancelled


def test_fifo_order_and_cancel():
    sem = FifoSemaphore(1)
    sem.acquire()
    order, cancel = [], threading.Event()

    def worker(i, ev=None):
        try:
            sem.acquire(ev)
            order.append(i)
            time.sleep(0.01)
            sem.release()
        except RequestCancelled:
            order.append(f"batal{i}")

    threads = []
    for i in range(6):
        t = threading.Thread(target=worker, args=(i, cancel if i == 2 else None))
        t.start()
        threads.append(t)
        time.sleep(0.02)  # pastikan urutan kedatangan 0..5
    cancel.set()
    time.sleep(0.4)  # penunggu #2 melihat flag batal saat masih antre
    sem.release()
    for t in threads:
        t.join(timeout=5)
    assert order == ["batal2", 0, 1, 3, 4, 5]
    assert sem.queued() == 0
    sem.acquire()  # slot tidak bocor: masih bisa diambil lagi
    sem.release()


def test_cancelled_before_acquire_raises():
    sem = FifoSemaphore(2)
    ev = threading.Event()
    ev.set()
    with pytest.raises(RequestCancelled):
        sem.acquire(ev)
