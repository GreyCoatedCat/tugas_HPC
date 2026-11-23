"""Indeks leksikal BM25 (Okapi) berbasis matriks sparse.

Bobot BM25 per (dokumen, term) dihitung sekali saat build dan disimpan dalam
matriks CSC [N_doc x V]. Saat kueri, skor dokumen = jumlah bobot pada kolom
term kueri -> akumulasi posting list (baseline: sekuensial, 1 thread).
"""

from __future__ import annotations

import json
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from .topk import topk


class BM25Index:
    def __init__(
        self,
        matrix: sp.csc_matrix,
        vocab: dict[str, int],
        idf: np.ndarray,
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.matrix = matrix
        self.vocab = vocab
        self.idf = idf
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

    @property
    def n_docs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.matrix.nnz)

    @classmethod
    def build(cls, token_lists: Iterable[list[str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        vocab: dict[str, int] = {}
        indptr = array("q", [0])
        indices = array("i")
        tfs = array("f")
        doc_len = array("f")
        for toks in token_lists:
            for term, cnt in Counter(toks).items():
                j = vocab.get(term)
                if j is None:
                    j = vocab[term] = len(vocab)
                indices.append(j)
                tfs.append(cnt)
            indptr.append(len(indices))
            doc_len.append(len(toks))

        n, v = len(doc_len), len(vocab)
        ind = np.frombuffer(indices, dtype=np.int32) if len(indices) else np.empty(0, np.int32)
        tf = np.frombuffer(tfs, dtype=np.float32) if len(tfs) else np.empty(0, np.float32)
        ptr = np.frombuffer(indptr, dtype=np.int64)
        dl = np.frombuffer(doc_len, dtype=np.float32) if n else np.empty(0, np.float32)
        avgdl = float(dl.mean()) if n and dl.sum() > 0 else 1.0

        df = np.bincount(ind, minlength=v).astype(np.float32)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)

        # w = idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)) — dihitung in-place hemat memori
        norm = (k1 * (1.0 - b + b * dl / avgdl)).astype(np.float32)
        denom = np.repeat(norm, np.diff(ptr))
        denom += tf
        w = tf * np.float32(k1 + 1.0)
        w /= denom
        del denom
        w *= idf[ind]
        csr = sp.csr_matrix((w, ind.copy(), ptr.copy()), shape=(n, v))
        del w
        return cls(csr.tocsc(), vocab, idf, avgdl, k1, b)

    def search(self, tokens: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        cols = sorted({self.vocab[t] for t in tokens if t in self.vocab})
        if not cols or self.n_docs == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        m = self.matrix
        starts = m.indptr[cols]
        ends = m.indptr[np.asarray(cols) + 1]
        rows = np.concatenate([m.indices[s:e] for s, e in zip(starts, ends)])
        vals = np.concatenate([m.data[s:e] for s, e in zip(starts, ends)])
        scores = np.bincount(rows, weights=vals, minlength=self.n_docs).astype(np.float32)
        return topk(scores, k, positive_only=True)

    # ---------------------------------------------------------------- I/O
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        m = self.matrix
        tmp = directory / "bm25.tmp.npz"
        np.savez(
            tmp,
            data=m.data,
            indices=m.indices,
            indptr=m.indptr,
            shape=np.asarray(m.shape, dtype=np.int64),
            idf=self.idf,
            params=np.asarray([self.avgdl, self.k1, self.b], dtype=np.float64),
        )
        tmp.replace(directory / "bm25.npz")
        terms = [None] * len(self.vocab)
        for t, i in self.vocab.items():
            terms[i] = t
        tmpv = directory / "bm25_vocab.tmp.json"
        tmpv.write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
        tmpv.replace(directory / "bm25_vocab.json")

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        directory = Path(directory)
        z = np.load(directory / "bm25.npz")
        shape = tuple(int(x) for x in z["shape"])
        matrix = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
        terms = json.loads((directory / "bm25_vocab.json").read_text(encoding="utf-8"))
        avgdl, k1, b = (float(x) for x in z["params"])
        return cls(matrix, {t: i for i, t in enumerate(terms)}, z["idf"], avgdl, k1, b)
