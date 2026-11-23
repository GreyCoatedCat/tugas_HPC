"""Metrik runtime: request yang sedang berjalan, latensi rolling, utilisasi CPU/GPU."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def percentile(values, q: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


class MetricsTracker:
    def __init__(self, maxlen: int = 20000) -> None:
        self._lock = threading.Lock()
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.in_flight = 0
        self.in_flight_by_mode: dict[str, int] = {}
        self.total = 0
        self.started_at = time.time()

    def start(self, mode: str | None = None) -> None:
        with self._lock:
            self.in_flight += 1
            if mode:
                self.in_flight_by_mode[mode] = self.in_flight_by_mode.get(mode, 0) + 1

    def finish(self, rec: dict[str, Any]) -> None:
        rec = {**rec, "t_end": time.time()}
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            m = rec.get("mode")
            if m in self.in_flight_by_mode:
                self.in_flight_by_mode[m] = max(0, self.in_flight_by_mode[m] - 1)
            self.total += 1
            self._records.append(rec)

    def snapshot(self, window_s: float = 60.0, mode: str | None = None) -> dict[str, Any]:
        """Ringkasan jendela waktu terakhir; ``mode`` membatasi ke satu mode pipeline."""
        now = time.time()
        with self._lock:
            recs = [r for r in self._records if now - r["t_end"] <= window_s and (mode is None or r.get("mode") == mode)]
            in_flight = self.in_flight if mode is None else self.in_flight_by_mode.get(mode, 0)
            total = self.total
        ok = [r for r in recs if not r.get("error")]
        stage_keys: set[str] = set()
        for r in ok:
            stage_keys.update((r.get("stages") or {}).keys())
        stage_mean = {
            k: round(float(np.mean([r["stages"].get(k, 0.0) for r in ok])), 2) for k in sorted(stage_keys)
        }
        e2e = [r.get("e2e_ms") for r in ok]
        ttft = [r.get("ttft_ms") for r in ok]
        span = min(window_s, now - self.started_at) or 1.0
        rnd = lambda x: None if x is None else round(x, 1)  # noqa: E731
        return {
            "window_s": window_s,
            "in_flight": in_flight,
            "total_requests": total,
            "completed": len(recs),
            "errors": len(recs) - len(ok),
            "refused": sum(1 for r in ok if r.get("refused")),
            "qps": round(len(ok) / span, 3),
            "e2e_p50_ms": rnd(percentile(e2e, 50)),
            "e2e_p95_ms": rnd(percentile(e2e, 95)),
            "ttft_p50_ms": rnd(percentile(ttft, 50)),
            "ttft_p95_ms": rnd(percentile(ttft, 95)),
            "stage_mean_ms": stage_mean,
        }


class ResourceSampler(threading.Thread):
    """Sampling utilisasi CPU (psutil) dan GPU (NVML) secara periodik."""

    def __init__(self, interval_s: float = 1.0, history: int = 600) -> None:
        super().__init__(daemon=True, name="resource-sampler")
        self.interval_s = interval_s
        self.history: deque[dict[str, Any]] = deque(maxlen=history)
        self._stop_evt = threading.Event()
        self._nvml = None
        self._handles: list = []

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:  # noqa: BLE001 - GPU opsional
            self._nvml = None

    def sample(self) -> dict[str, Any]:
        import psutil

        s: dict[str, Any] = {
            "t": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
            "ram_used_gb": round(psutil.virtual_memory().used / 2**30, 2),
        }
        if self._nvml:
            gpus = []
            for h in self._handles:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(h)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(h)
                    gpus.append(
                        {
                            "util_percent": util.gpu,
                            "mem_util_percent": util.memory,
                            "mem_used_mb": round(mem.used / 2**20),
                            "mem_total_mb": round(mem.total / 2**20),
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
            s["gpus"] = gpus
        return s

    def run(self) -> None:
        import psutil

        self._init_nvml()
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        while not self._stop_evt.wait(self.interval_s):
            try:
                self.history.append(self.sample())
            except Exception:  # noqa: BLE001
                pass

    def latest(self) -> dict[str, Any] | None:
        return self.history[-1] if self.history else None

    def summary(self, window_s: float = 60.0) -> dict[str, Any]:
        now = time.time()
        recent = [s for s in list(self.history) if now - s["t"] <= window_s]
        if not recent:
            return {}
        out: dict[str, Any] = {"cpu_percent_mean": round(float(np.mean([s["cpu_percent"] for s in recent])), 1)}
        if recent[-1].get("gpus"):
            out["gpu_util_mean"] = round(float(np.mean([s["gpus"][0]["util_percent"] for s in recent if s.get("gpus")])), 1)
            out["gpu_mem_used_mb_max"] = max(s["gpus"][0]["mem_used_mb"] for s in recent if s.get("gpus"))
        return out

    def stop(self) -> None:
        self._stop_evt.set()


class RequestLogger:
    """Tulis timing setiap request ke JSONL (untuk analisis profiling offline)."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, rec: dict[str, Any]) -> None:
        path = self.dir / f"requests-{datetime.now():%Y%m%d}.jsonl"
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
