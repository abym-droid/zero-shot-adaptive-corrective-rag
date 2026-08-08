"""ColBERTv2 late-interaction retriever (optional dependency).

The highest-fidelity (and most expensive) retrieval option, intended for
hard queries once experiments reach that ablation. `ragatouille` drags in
the full ColBERT training stack, so it is deliberately NOT in the base env:
the import is guarded inside methods so this module imports cleanly
everywhere, and every entry point raises an informative ImportError when it
is absent. The API usage below follows ragatouille's documented surface
(``from_pretrained`` / ``.index`` / ``.from_index`` / ``.search``) and
should get a smoke test once the optional dependency is actually installed.
"""
from __future__ import annotations

from pathlib import Path

from adarag.config import settings
from adarag.retrieval.base import Doc, Retriever

_INSTALL_HINT = (
    "ColBERTRetriever requires the OPTIONAL 'ragatouille' package, which is "
    "not installed in this environment. It is intentionally excluded from "
    "the base env (heavy ColBERT stack). Install it with:\n"
    "    pip install ragatouille\n"
    "(or: pip install -r requirements-optional.txt). BM25Retriever and "
    "DenseRetriever cover the baseline tiers without it."
)


def _require_ragatouille():
    """Import and return `RAGPretrainedModel`, or raise with install
    instructions."""
    try:
        from ragatouille import RAGPretrainedModel
    except ImportError as exc:  # pragma: no cover - env without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return RAGPretrainedModel


class ColBERTRetriever(Retriever):
    """Late-interaction retrieval via ragatouille / ColBERTv2.

    Build with `ColBERTRetriever.build(chunks, index_dir)` (indexes on disk -
    ColBERT indices are inherently on-disk artefacts), restore with
    `ColBERTRetriever.load(index_dir)`.
    """

    name = "colbert"

    def __init__(self, model, chunk_meta: dict[str, dict] | None = None):
        """Wrap a ragatouille model with a loaded index; prefer `build()` /
        `load()`. ``chunk_meta`` maps chunk_id -> chunk dict so results get
        their title/provenance metadata back."""
        self._model = model
        self._chunk_meta = chunk_meta or {}

    @classmethod
    def build(
        cls,
        chunks: list[dict],
        index_dir: str | Path,
        model_name: str = settings.colbert_model,
        index_name: str = "adarag",
    ) -> "ColBERTRetriever":
        """Encode and index corpus chunks with ColBERTv2 (writes the on-disk
        index under ``index_dir``)."""
        if not chunks:
            raise ValueError("ColBERTRetriever.build: chunks list is empty")
        RAGPretrainedModel = _require_ragatouille()
        model = RAGPretrainedModel.from_pretrained(model_name, index_root=str(index_dir))
        model.index(
            collection=[c["text"] for c in chunks],
            document_ids=[c["chunk_id"] for c in chunks],
            index_name=index_name,
        )
        return cls(model, {c["chunk_id"]: c for c in chunks})

    @classmethod
    def load(cls, index_path: str | Path) -> "ColBERTRetriever":
        """Open an existing on-disk ColBERT index
        (``<index_dir>/colbert/indexes/<index_name>``, as written by
        `build`)."""
        RAGPretrainedModel = _require_ragatouille()
        return cls(RAGPretrainedModel.from_index(str(index_path)))

    def search(self, query: str, k: int = 5) -> list[Doc]:
        """Late-interaction (MaxSim) top-``k`` search, best first."""
        hits = self._model.search(query=query, k=k)
        docs: list[Doc] = []
        for hit in hits:
            chunk_id = str(hit.get("document_id") or hit.get("passage_id", ""))
            meta_src = self._chunk_meta.get(chunk_id, {})
            docs.append(
                Doc(
                    doc_id=chunk_id,
                    text=hit.get("content", ""),
                    score=float(hit.get("score", 0.0)),
                    meta={
                        "chunk_id": chunk_id,
                        "parent_doc_id": meta_src.get("doc_id", ""),
                        "title": meta_src.get("title", ""),
                        "rank": int(hit.get("rank", 0)),
                    },
                )
            )
        return docs
