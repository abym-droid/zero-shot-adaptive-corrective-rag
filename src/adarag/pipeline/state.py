"""Shared LangGraph state for the adaptive RAG pipeline.

Every node reads from and writes to one ``RAGState`` dict; the ``trace``
channel accumulates per-node telemetry (latency, token counts) so
cost/latency metrics come from a single run record. Stdlib-only imports so
the module loads anywhere with zero heavy dependencies.
"""
from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

if TYPE_CHECKING:  # import only for type checkers; avoids runtime coupling
    from adarag.retrieval.base import Doc


class RAGState(TypedDict, total=False):
    """Mutable shared state threaded through the LangGraph nodes.

    ``tier`` holds the ``RetrievalTier`` *value* string; ``force_min_tier``
    is the escalation hint from the gate->route back-edge (router must not
    pick a tier below it); ``docs`` are serialized ``Doc`` dicts. ``trace``
    is annotated with ``operator.add`` so each node contributes only its new
    entries and LangGraph concatenates them.
    """

    question: str
    tier: str
    force_min_tier: str | None
    route_reason: str
    docs: list[dict]
    answer: str
    verdict: str | None
    gate_reason: str
    escalations: int
    trace: Annotated[list[dict], operator.add]


def trace_entry(
    node: str,
    latency_s: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    detail: dict[str, Any] | None = None,
) -> dict:
    """Build one telemetry record for the ``trace`` channel; ``detail`` is an
    optional node-specific payload (chosen tier, verdict, per-step breakdown).
    """
    return {
        "node": node,
        "latency_s": round(latency_s, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "detail": detail or {},
    }


def serialize_docs(docs: list[Doc]) -> list[dict]:
    """Serialize ``Doc`` objects into plain dicts for storage in ``RAGState``.

    Duck-typed (dataclass ``asdict`` with attribute-access fallback) so this
    module carries no runtime import of the retrieval package.
    """
    out: list[dict] = []
    for d in docs:
        if dataclasses.is_dataclass(d) and not isinstance(d, type):
            out.append(dataclasses.asdict(d))
        else:  # tolerate any object exposing the Doc attributes
            out.append(
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "score": float(d.score),
                    "meta": dict(d.meta),
                }
            )
    return out


def deserialize_docs(doc_dicts: list[dict]) -> list["Doc"]:
    """Rehydrate serialized doc dicts back into ``Doc`` objects; ``Doc`` is
    imported lazily so this module works where the retrieval extras are absent.
    """
    from adarag.retrieval.base import Doc

    return [
        Doc(
            doc_id=d["doc_id"],
            text=d["text"],
            score=d.get("score", 0.0),
            meta=d.get("meta", {}),
        )
        for d in doc_dicts
    ]
