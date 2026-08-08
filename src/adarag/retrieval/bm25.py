"""Sparse lexical retriever (BM25 baseline).

BM25 is the cheap sparse option for single-step retrieval; it is also the
offline-friendly retriever the test suite uses (no model downloads). Backed
by `rank_bm25.BM25Okapi`. Persistence is a single pickle - the whole index
lives comfortably in RAM for every corpus in scope.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from adarag.retrieval.base import Doc, Retriever, chunk_to_doc, simple_tokenize

_PICKLE_NAME = "bm25.pkl"
_FORMAT_VERSION = 1


class BM25Retriever(Retriever):
    """BM25Okapi over corpus chunks, with pickle save/load.

    Build with `BM25Retriever.build(chunks)`, persist with `save(dir)`,
    restore with `BM25Retriever.load(dir)`.
    """

    name = "bm25"

    def __init__(self, bm25, chunks: list[dict]):
        """Wrap a fitted BM25 model; prefer `build()` / `load()`. ``chunks``
        must be in the same order the model was fitted on."""
        self._bm25 = bm25
        self._chunks = chunks

    def __len__(self) -> int:
        return len(self._chunks)

    @classmethod
    def build(cls, chunks: list[dict]) -> "BM25Retriever":
        """Fit BM25 over chunk texts (dicts from
        `adarag.corpus.chunk.chunk_corpus`)."""
        if not chunks:
            raise ValueError("BM25Retriever.build: chunks list is empty")
        from rank_bm25 import BM25Okapi  # cheap, but keep imports local & symmetric

        tokenized = [simple_tokenize(c["text"]) for c in chunks]
        return cls(BM25Okapi(tokenized), list(chunks))

    def search(self, query: str, k: int = 5) -> list[Doc]:
        """Rank all chunks by BM25 score and return the top ``k``, best
        first (may be shorter than ``k``)."""
        tokens = simple_tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        k = min(k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [chunk_to_doc(self._chunks[i], float(scores[i])) for i in top]

    def save(self, out_dir: str | Path) -> Path:
        """Persist the fitted model + chunks as one pickle; returns the
        pickle path (``out_dir`` is created if missing)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "bm25": self._bm25,
            "chunks": self._chunks,
        }
        path = out_dir / _PICKLE_NAME
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, in_dir: str | Path) -> "BM25Retriever":
        """Restore a retriever previously written by `save`."""
        path = Path(in_dir) / _PICKLE_NAME
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(payload["bm25"], payload["chunks"])
