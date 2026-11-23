"""Thread inferensi khusus untuk model PyTorch.

PyTorch membuat cuBLAS handle + workspace (puluhan MB VRAM) untuk setiap thread
yang menjalankan kernel GPU, dan workspace itu tidak dibebaskan selama proses
hidup. Server melayani request dari thread pool besar (hingga ratusan thread),
sehingga tanpa koordinasi VRAM terus bertambah mengikuti jumlah thread sampai
`cublasCreate` gagal (CUBLAS_STATUS_ALLOC_FAILED) pada 50+ pengguna konkuren.

Karena itu setiap model menjalankan forward pass-nya di satu thread tetap:
jumlah handle konstan, dan pemakaian model otomatis sekuensial (tokenizer HF
fast juga tidak aman dipakai paralel). Paralelisme/batching untuk mode
optimized ditambahkan di level ini atau di modul C++.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class InferenceWorker:
    def __init__(self, name: str) -> None:
        self._prefix = f"infer-{name}"
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._prefix)

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if threading.current_thread().name.startswith(self._prefix):  # pemanggilan bersarang
            return fn(*args, **kwargs)
        return self._exec.submit(fn, *args, **kwargs).result()

    def close(self) -> None:
        self._exec.shutdown(wait=False, cancel_futures=True)
