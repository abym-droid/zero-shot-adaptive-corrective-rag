"""Tests for the bounded escalation ladder through the full graph.

Pins the one-promotion bound: on an "incorrect" gate verdict the query
escalates to the NEXT tier only (A→B, B→C, never skipping), at most once
per query (settings.max_escalations = 1), and nothing escalates above
tier C. All asserted on the final RAGState from AdaptiveRAG.answer,
driven entirely by FakeBackend.
"""
from __future__ import annotations

from adarag.config import RetrievalTier
from adarag.llm.fake import FakeBackend
from adarag.pipeline.graph import AdaptiveRAG

ANSWERS = ["ANSWER: Paris"] * 10  # generous: serves tiers A/B and IRCoT steps


def _rag(bm25_retriever, router_responses, gate_responses):
    return AdaptiveRAG(
        backend=FakeBackend(ANSWERS),
        retriever=bm25_retriever,
        gate_backend=FakeBackend(gate_responses),
        router_backend=FakeBackend(router_responses),
    )


def test_incorrect_at_b_escalates_to_c_then_stops(
    bm25_retriever, router_json, gate_json
):
    # B → gate says incorrect → escalate to C → gate says incorrect again
    # → STOP: bound reached and C has no next tier. Exactly one escalation.
    rag = _rag(
        bm25_retriever,
        router_responses=[
            router_json(RetrievalTier.SINGLE_STEP.value),
            router_json(RetrievalTier.SINGLE_STEP.value),  # promoted by hint
        ],
        gate_responses=[gate_json("incorrect"), gate_json("incorrect")],
    )
    state = rag.answer("What is the capital of France?")
    assert state["escalations"] == 1
    assert state["tier"] == RetrievalTier.ITERATIVE.value
    assert state["verdict"] == "incorrect"  # final verdict stands, answer kept
    assert state["answer"]


def test_incorrect_at_a_escalates_to_b(bm25_retriever, router_json, gate_json):
    rag = _rag(
        bm25_retriever,
        router_responses=[
            router_json(RetrievalTier.NO_RETRIEVAL.value),
            router_json(RetrievalTier.NO_RETRIEVAL.value),  # promoted by hint
        ],
        gate_responses=[gate_json("incorrect"), gate_json("correct")],
    )
    state = rag.answer("What is the capital of France?")
    assert state["escalations"] == 1
    assert state["tier"] == RetrievalTier.SINGLE_STEP.value
    assert state["verdict"] == "correct"
    # After escalating into a retrieval tier, evidence must be present.
    assert state["docs"]


def test_incorrect_at_top_tier_c_does_not_escalate(
    bm25_retriever, router_json, gate_json
):
    # ESCALATION_NEXT[ITERATIVE] is None: answer stands, zero escalations.
    rag = _rag(
        bm25_retriever,
        router_responses=[router_json(RetrievalTier.ITERATIVE.value)],
        gate_responses=[gate_json("incorrect")],
    )
    state = rag.answer("Hard multi-hop question?")
    assert state.get("escalations", 0) == 0
    assert state["tier"] == RetrievalTier.ITERATIVE.value
    assert state["verdict"] == "incorrect"
    assert state["answer"]


def test_correct_verdict_never_escalates(bm25_retriever, router_json, gate_json):
    rag = _rag(
        bm25_retriever,
        router_responses=[router_json(RetrievalTier.SINGLE_STEP.value)],
        gate_responses=[gate_json("correct")],
    )
    state = rag.answer("What is the capital of France?")
    assert state.get("escalations", 0) == 0
    assert state["tier"] == RetrievalTier.SINGLE_STEP.value


def test_ambiguous_verdict_proceeds_without_escalation(
    bm25_retriever, router_json, gate_json
):
    # Ambiguous is CRAG-lite territory (doc filtering), never an escalation.
    rag = _rag(
        bm25_retriever,
        router_responses=[router_json(RetrievalTier.SINGLE_STEP.value)],
        gate_responses=[gate_json("ambiguous", useful_doc_ids=["paris-0"])],
    )
    state = rag.answer("What is the capital of France?")
    assert state.get("escalations", 0) == 0
    assert state["verdict"] == "ambiguous"
    assert state["answer"]
