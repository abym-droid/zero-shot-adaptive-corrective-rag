"""Shared fixtures for the offline test suite - no network, no model
downloads, no GPU. LLM calls go through FakeBackend with canned responses,
and a tiny six-chunk rank-bm25 corpus stands in for real retrieval.

adarag imports happen inside the fixtures on purpose: a missing module
only fails the tests that need it instead of breaking collection.
"""
from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Tiny corpus: six topically disjoint chunks so BM25 ranking is unambiguous.
# Shape matches corpus/chunk.py output: {"chunk_id", "doc_id", "title", "text"}.
# ---------------------------------------------------------------------------
TINY_CHUNKS: list[dict] = [
    {
        "chunk_id": "paris-0",
        "doc_id": "paris",
        "title": "Paris",
        "text": (
            "Paris is the capital of France and its most populous city. "
            "The city sits on the river Seine in northern France."
        ),
    },
    {
        "chunk_id": "eiffel-0",
        "doc_id": "eiffel",
        "title": "Eiffel Tower",
        "text": (
            "The Eiffel Tower is a wrought-iron lattice tower built by "
            "Gustave Eiffel for the 1889 World's Fair in Paris."
        ),
    },
    {
        "chunk_id": "5g-0",
        "doc_id": "5g",
        "title": "5G NR",
        "text": (
            "5G New Radio is the radio access technology standardised by "
            "3GPP for fifth generation mobile telecom networks."
        ),
    },
    {
        "chunk_id": "photo-0",
        "doc_id": "photo",
        "title": "Photosynthesis",
        "text": (
            "Photosynthesis is the process by which green plants convert "
            "sunlight, water and carbon dioxide into glucose and oxygen."
        ),
    },
    {
        "chunk_id": "amazon-0",
        "doc_id": "amazon",
        "title": "Amazon River",
        "text": (
            "The Amazon river in South America is the largest river in "
            "the world by discharge volume of water."
        ),
    },
    {
        "chunk_id": "insulin-0",
        "doc_id": "insulin",
        "title": "Insulin",
        "text": (
            "Insulin is a peptide hormone produced by the pancreas that "
            "regulates blood glucose; its deficiency causes diabetes."
        ),
    },
]


@pytest.fixture()
def tiny_chunks() -> list[dict]:
    """Fresh copy of the six-chunk in-memory corpus (mutation-safe)."""
    return [dict(c) for c in TINY_CHUNKS]


@pytest.fixture()
def bm25_retriever(tiny_chunks):
    """A BM25Retriever built in-memory over the tiny corpus (no disk I/O)."""
    from adarag.retrieval.bm25 import BM25Retriever

    return BM25Retriever.build(tiny_chunks)


# ---------------------------------------------------------------------------
# JSON-response factories for scripting FakeBackend replies.
# ---------------------------------------------------------------------------
@pytest.fixture()
def router_json():
    """Factory producing a router-schema JSON string for a given tier."""

    def _make(tier: str, reason: str = "test-reason") -> str:
        return json.dumps({"tier": tier, "reason": reason})

    return _make


@pytest.fixture()
def gate_json():
    """Factory producing a gate-schema JSON string for a given verdict."""

    def _make(
        verdict: str,
        reason: str = "test-reason",
        useful_doc_ids: list[str] | None = None,
    ) -> str:
        return json.dumps(
            {
                "verdict": verdict,
                "reason": reason,
                "useful_doc_ids": useful_doc_ids or [],
            }
        )

    return _make
