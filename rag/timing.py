"""Pencatat waktu per tahap pipeline (milidetik)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class StageTimer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + ms

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def as_dict(self, ndigits: int = 3) -> dict[str, float]:
        return {k: round(v, ndigits) for k, v in self.stages.items()}
