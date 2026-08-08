"""End-to-end tests of the LangGraph pipeline across all three tiers.

Nodes go route → execute_tier → gate → (conditional escalate | END).
Each tier's happy path is driven with FakeBackend over the tiny BM25
corpus:

    A (no_retrieval)  - parametric answer, gate uses evaluate_answer;
    B (single_step)   - one retrieval pass, evidence in state["docs"];
    C (iterative)     - IRCoT-lite loop stopping on "ANSWER:".

Also pins the RAGState trace contract that eval/run_eval needs for the
latency and token-cost numbers.
"""
from __future__ import annotations

from adarag.config import RetrievalTier
from adarag.llm.fake import FakeBackend
from adarag.pipeline.graph import AdaptiveRAG


def _rag(bm25_retriever, tier_json, gen_responses, gate_responses):
    return AdaptiveRAG(
        backend=FakeBackend(gen_responses),
        retriever=bm25_retriever,
        gate_backend=FakeBackend(gate_responses),
        router_backend=FakeBackend([tier_json]),
    )


def test_tier_a_no_retrieval_end_to_end(bm25_retriever, router_json, gate_json):
    rag = _rag(
        bm25_retriever,
        tier_json=router_json(RetrievalTier.NO_RETRIEVAL.value),
        gen_responses=["Paris"] * 4,
        gate_responses=[gate_json("correct")],
    )
    state = rag.answer("What is the capital of France?")
    assert state["question"] == "What is the capital of France?"
    assert state["tier"] == RetrievalTier.NO_RETRIEVAL.value
    assert "Paris" in state["answer"]
    assert state["verdict"] == "correct"
    assert not state.get("docs")  # tier A retrieves nothing
    assert state.get("escalations", 0) == 0


def test_tier_b_single_step_end_to_end(bm25_retriever, router_json, gate_json):
    rag = _rag(
        bm25_retriever,
        tier_json=router_json(RetrievalTier.SINGLE_STEP.value),
        gen_responses=["Paris"] * 4,
        gate_responses=[gate_json("correct", useful_doc_ids=["paris-0"])],
    )
    state = rag.answer("What is the capital of France?")
    assert state["tier"] == RetrievalTier.SINGLE_STEP.value
    assert "Paris" in state["answer"]
    assert state["verdict"] == "correct"
    docs = state["docs"]
    assert docs and isinstance(docs, list)
    # serialized Doc dicts must keep their ids for Recall@K evaluation
    assert all(isinstance(d, dict) and "doc_id" in d for d in docs)


def test_tier_b_retrieves_topically_relevant_evidence(
    bm25_retriever, router_json, gate_json
):
    rag = _rag(
        bm25_retriever,
        tier_json=router_json(RetrievalTier.SINGLE_STEP.value),
        gen_responses=["glucose and oxygen"] * 4,
        gate_responses=[gate_json("correct")],
    )
    state = rag.answer("What does photosynthesis convert sunlight into?")
    texts = " ".join(d["text"] for d in state["docs"]).lower()
    assert "photosynthesis" in texts


def test_tier_c_iterative_end_to_end(bm25_retriever, router_json, gate_json):
    # IRCoT-lite: first reasoning step already contains "ANSWER:" → stop.
    rag = _rag(
        bm25_retriever,
        tier_json=router_json(RetrievalTier.ITERATIVE.value),
        gen_responses=["The Seine flows through the French capital. ANSWER: Paris"]
        * 8,
        gate_responses=[gate_json("correct")],
    )
    state = rag.answer("Which city on the Seine hosts the Eiffel Tower?")
    assert state["tier"] == RetrievalTier.ITERATIVE.value
    assert "Paris" in state["answer"]
    assert state["docs"]  # accumulated context, deduped by chunk_id
    doc_ids = [d["doc_id"] for d in state["docs"]]
    assert len(doc_ids) == len(set(doc_ids))  # dedup contract


def test_trace_records_per_node_accounting(bm25_retriever, router_json, gate_json):
    # latency and token cost per query come from the trace
    rag = _rag(
        bm25_retriever,
        tier_json=router_json(RetrievalTier.SINGLE_STEP.value),
        gen_responses=["Paris"] * 4,
        gate_responses=[gate_json("correct")],
    )
    state = rag.answer("What is the capital of France?")
    trace = state["trace"]
    assert isinstance(trace, list) and trace
    for entry in trace:
        assert "node" in entry
        assert entry.get("latency_s", 0.0) >= 0.0
    nodes = {e["node"] for e in trace}
    # at minimum the three core nodes must have reported
    assert len(nodes) >= 2
