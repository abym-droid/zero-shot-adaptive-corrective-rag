"""Corpus preparation: chunking and index building.

Turns downloaded corpus jsonl files into the chunk dicts and on-disk
BM25 / FAISS indices consumed by `adarag.retrieval`.
"""
from __future__ import annotations

from adarag.corpus.build import build_indices
from adarag.corpus.chunk import chunk_corpus, chunk_text

__all__ = ["chunk_text", "chunk_corpus", "build_indices"]
