"""Retrieval backends for the adaptive-RAG pipeline.

Exports the shared `Doc` / `Retriever` interface plus the three concrete
retrievers: BM25 (sparse baseline), dense bi-encoder + FAISS, and the
optional ColBERTv2 late-interaction retriever. `load_retriever` is a
convenience for opening a saved index directory by kind.
"""
from __future__ import annotations

from pathlib import Path

from adarag.retrieval.base import Doc, Retriever, simple_tokenize
from adarag.retrieval.bm25 import BM25Retriever
from adarag.retrieval.colbert import ColBERTRetriever
from adarag.retrieval.dense import DenseRetriever

__all__ = [
    "Doc",
    "Retriever",
    "simple_tokenize",
    "BM25Retriever",
    "DenseRetriever",
    "ColBERTRetriever",
    "load_retriever",
]

_KIND_DIRS = {"bm25": "bm25", "dense": "dense"}


def load_retriever(index_dir: str | Path, kind: str = "bm25", **kwargs) -> Retriever:
    """Open a saved retriever from an ``indices/<name>/`` directory.

    ``index_dir`` is the per-corpus index root written by
    `adarag.corpus.build.build_indices` (containing ``bm25/`` and/or
    ``dense/`` subdirectories), or a direct path to one of those
    subdirectories. Extra kwargs are forwarded to the backend's ``load``
    (e.g. ``encoder=``, ``device=`` for dense).
    """
    index_dir = Path(index_dir)
    if kind not in _KIND_DIRS:
        raise ValueError(f"Unknown retriever kind {kind!r}; expected one of {sorted(_KIND_DIRS)}")
    sub = index_dir / _KIND_DIRS[kind]
    target = sub if sub.is_dir() else index_dir
    if kind == "bm25":
        return BM25Retriever.load(target, **kwargs)
    return DenseRetriever.load(target, **kwargs)
