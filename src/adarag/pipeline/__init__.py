"""Adaptive RAG pipeline package.

Bundles the LangGraph orchestrator (route -> execute_tier -> gate -> bounded
escalate back-edge), the shared graph state, and the three tier runners.
"""
from adarag.pipeline.graph import AdaptiveRAG
from adarag.pipeline.state import RAGState, deserialize_docs, serialize_docs, trace_entry
from adarag.pipeline.tiers import (
    TierResult,
    run_iterative,
    run_no_retrieval,
    run_single_step,
)

__all__ = [
    "AdaptiveRAG",
    "RAGState",
    "TierResult",
    "deserialize_docs",
    "run_iterative",
    "run_no_retrieval",
    "run_single_step",
    "serialize_docs",
    "trace_entry",
]
