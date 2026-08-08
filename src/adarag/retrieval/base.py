"""Retrieval interface shared by every retriever backend.

The single-step and iterative tiers call a `Retriever` through the uniform
`search()` interface, so the router and tier executors never need to know
which index family is behind it. The `Doc` dataclass is the single retrieval
result type used across the system (gate evaluation, context assembly, trace
serialization).
"""
from __future__ import annotations

import abc
import dataclasses
import re
from dataclasses import dataclass, field


@dataclass
class Doc:
    """One retrieved chunk.

    ``doc_id`` is the chunk id for chunked corpora (e.g.
    ``"3gpp_38.331::c17"``) - the id the iterative tier dedupes on.
    ``score`` is retriever-specific but always higher-is-better (BM25,
    inner product, or ColBERT maxsim). For corpus chunks ``meta`` carries
    ``{"chunk_id", "parent_doc_id", "title"}``.
    """

    doc_id: str
    text: str
    score: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for `RAGState.docs` / trace jsonl)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Doc":
        """Rebuild a `Doc` from `to_dict()` output."""
        return cls(
            doc_id=d["doc_id"],
            text=d["text"],
            score=float(d.get("score", 0.0)),
            meta=dict(d.get("meta", {})),
        )


class Retriever(abc.ABC):
    """Abstract retriever: query in, ranked `Doc` list out.

    Concrete implementations: `BM25Retriever` (sparse baseline),
    `DenseRetriever` (bi-encoder + FAISS), `ColBERTRetriever`
    (late interaction, optional dependency).
    """

    #: Short human-readable backend name (used in traces and run summaries).
    name: str = "abstract"

    @abc.abstractmethod
    def search(self, query: str, k: int = 5) -> list[Doc]:
        """Return the top-``k`` chunks for ``query``, highest score first
        (fewer when the index holds fewer chunks)."""


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase + regex word tokenizer shared by sparse retrievers.

    Deliberately minimal (no stemming/stopwords) so index builds are fast,
    deterministic and dependency-free - BM25 here is a baseline, not a
    contribution.
    """
    return _TOKEN_RE.findall(text.lower())


def chunk_to_doc(chunk: dict, score: float) -> Doc:
    """Convert a corpus chunk dict (`adarag.corpus.chunk`) into a `Doc`,
    with ``doc_id`` set to the chunk id and provenance in ``meta``."""
    return Doc(
        doc_id=chunk["chunk_id"],
        text=chunk["text"],
        score=score,
        meta={
            "chunk_id": chunk["chunk_id"],
            "parent_doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
        },
    )
