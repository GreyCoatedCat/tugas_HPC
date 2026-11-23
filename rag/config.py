"""Pemuatan konfigurasi (config.yaml + config.local.yaml opsional)."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
_CACHE: dict[str, dict] = {}


def _merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Baca config.yaml (atau $RAG_CONFIG) lalu merge config.local.yaml bila ada."""
    p = Path(path or os.environ.get("RAG_CONFIG") or ROOT / "config.yaml").resolve()
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    local = p.with_name(f"{p.stem}.local{p.suffix}")
    if local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    paths = cfg.setdefault("paths", {})
    for k, v in list(paths.items()):
        pp = Path(v)
        paths[k] = str(pp if pp.is_absolute() else (ROOT / pp))
    _CACHE[key] = cfg
    return cfg


def resolve_device(spec: str | None) -> str:
    if spec and spec != "auto":
        return spec
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ModeConfig:
    """Parameter eksekusi untuk satu mode (baseline / optimized / ...)."""

    name: str
    description: str = ""
    embed_batch_size: int = 1
    embed_dtype: str = "float32"
    vector_backend: str = "numpy"
    vector_threads: int = 1
    vector_dtype: str = "float32"         # penyimpanan matriks di backend GPU (float16 = hemat VRAM & bandwidth)
    bm25_backend: str = "scipy"
    bm25_threads: int = 1
    model_fp16: bool = False              # autocast FP16 untuk embedding kueri & reranker
    rerank_batch_size: int = 0            # 0 = pakai models.reranker.batch_size
    retrieval_batch_max: int = 1          # > 1 = dynamic batching permintaan konkuren
    retrieval_batch_window_ms: float = 0.0
    overlap_cpu_gpu: bool = False         # BM25 (CPU) berjalan bersamaan dengan pencarian vektor (GPU)
    ingest_workers: int = 1
    pipeline_max_concurrency: int = 1
    llm_max_concurrency: int = 1

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModeConfig":
        known = {f.name for f in fields(cls)}
        return cls(name=name, **{k: v for k, v in (d or {}).items() if k in known and k != "name"})


def mode_configs(cfg: dict) -> dict[str, ModeConfig]:
    return {name: ModeConfig.from_dict(name, d) for name, d in (cfg.get("modes") or {}).items()}


def chunk_params(cfg: dict) -> dict[str, int]:
    c = cfg.get("chunking") or {}
    return {
        "chunk_words": int(c.get("chunk_words", 180)),
        "overlap_words": int(c.get("overlap_words", 40)),
        "min_words": int(c.get("min_words", 20)),
    }
