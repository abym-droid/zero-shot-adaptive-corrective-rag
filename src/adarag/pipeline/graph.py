"""LangGraph orchestrator: route -> execute_tier -> gate -> (escalate | END).

The escalate back-edge is the headline novelty: on an ``incorrect`` verdict
the query is promoted to the next tier only (A->B or B->C, never skipping),
bounded by ``settings.max_escalations`` - set it to 0 for the no-escalation
ablation. Every node contributes a trace entry (latency, prompt/completion
tokens) so cost/latency metrics fall out of the run record.
"""
from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from adarag.config import ESCALATION_NEXT, RetrievalTier, Settings, settings as default_settings
from adarag.pipeline.state import RAGState, deserialize_docs, serialize_docs, trace_entry
from adarag.pipeline.tiers import run_iterative, run_no_retrieval, run_single_step

if TYPE_CHECKING:  # typing only - runtime binding happens lazily in __init__
    from langgraph.graph.state import CompiledStateGraph

    from adarag.llm.base import LLMBackend
    from adarag.retrieval.base import Retriever


class AdaptiveRAG:
    """Adaptive RAG pipeline: LLM routing + corrective escalation.

    Composes the zero-shot router, the three tier executors and the
    corrective gate into one LangGraph ``StateGraph`` over ``RAGState``,
    with the bounded gate->route escalation back-edge. Router and gate
    default to the generator ``backend`` unless given their own;
    ``settings.max_escalations`` bounds the back-edge (locked default 1;
    pass a copy with 0 for the ablation) and ``prompt_variant`` picks the
    router prompt.
    """

    NODE_ROUTE = "route"
    NODE_EXECUTE = "execute_tier"
    NODE_GATE = "gate"
    NODE_ESCALATE = "escalate"

    def __init__(
        self,
        backend: "LLMBackend",
        retriever: "Retriever",
        gate_backend: "LLMBackend | None" = None,
        router_backend: "LLMBackend | None" = None,
        settings: Settings = default_settings,
        prompt_variant: str = "v1",
    ) -> None:
        # Lazy imports: binding here (not at module import) keeps the pipeline
        # package importable in isolation and on minimal installs.
        from adarag.gate import CorrectiveGate
        from adarag.router import Router

        self.backend = backend
        self.retriever = retriever
        self.settings = settings
        self.router = Router(router_backend or backend, prompt_variant=prompt_variant)
        self.gate = CorrectiveGate(gate_backend or backend)
        self._compiled: "CompiledStateGraph | None" = None

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _route_node(self, state: RAGState) -> dict:
        """Route the question to a tier; honour the escalation hint."""
        t0 = perf_counter()
        forced_raw = state.get("force_min_tier")
        forced = RetrievalTier(forced_raw) if forced_raw else None
        decision, gen = self.router.route(state["question"], force_min_tier=forced)
        entry = trace_entry(
            self.NODE_ROUTE,
            perf_counter() - t0,
            gen.prompt_tokens,
            gen.completion_tokens,
            detail={
                "tier": decision.tier.value,
                "reason": decision.reason,
                "force_min_tier": forced_raw,
            },
        )
        return {
            "tier": decision.tier.value,
            "route_reason": decision.reason,
            "trace": [entry],
        }

    def _execute_node(self, state: RAGState) -> dict:
        """Run the tier chosen by the router (A / B / C)."""
        t0 = perf_counter()
        tier = RetrievalTier(state["tier"])
        if tier is RetrievalTier.NO_RETRIEVAL:
            result = run_no_retrieval(state, self.backend)
        elif tier is RetrievalTier.SINGLE_STEP:
            result = run_single_step(
                state, self.backend, self.retriever, self.settings.top_k
            )
        else:  # RetrievalTier.ITERATIVE
            result = run_iterative(
                state, self.backend, self.retriever, self.settings.top_k
            )
        entry = trace_entry(
            self.NODE_EXECUTE,
            perf_counter() - t0,
            sum(s["prompt_tokens"] for s in result.steps),
            sum(s["completion_tokens"] for s in result.steps),
            detail={"tier": tier.value, "n_docs": len(result.docs), "steps": result.steps},
        )
        return {
            "answer": result.answer,
            "docs": serialize_docs(result.docs),
            "trace": [entry],
        }

    def _gate_node(self, state: RAGState) -> dict:
        """CRAG-style verdict; tier A (or empty evidence) judges the answer."""
        t0 = perf_counter()
        tier = RetrievalTier(state["tier"])
        doc_dicts = state.get("docs") or []
        if tier is RetrievalTier.NO_RETRIEVAL or not doc_dicts:
            verdict, gen = self.gate.evaluate_answer(
                state["question"], state.get("answer", "")
            )
        else:
            verdict, gen = self.gate.evaluate(
                state["question"], deserialize_docs(doc_dicts)
            )
        updates: dict = {"verdict": verdict.verdict, "gate_reason": verdict.reason}
        kept_n = None
        # CRAG-lite refinement: on ambiguous, keep only gate-endorsed docs.
        if verdict.verdict == "ambiguous" and verdict.useful_doc_ids and doc_dicts:
            keep = set(verdict.useful_doc_ids)
            kept = [d for d in doc_dicts if d.get("doc_id") in keep]
            if kept:
                updates["docs"] = kept
                kept_n = len(kept)
        entry = trace_entry(
            self.NODE_GATE,
            perf_counter() - t0,
            gen.prompt_tokens,
            gen.completion_tokens,
            detail={
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                "useful_doc_ids": list(verdict.useful_doc_ids),
                "docs_kept": kept_n,
            },
        )
        updates["trace"] = [entry]
        return updates

    def _escalate_node(self, state: RAGState) -> dict:
        """Back-edge action: promote to the next tier only (A->B or B->C)."""
        t0 = perf_counter()
        current = RetrievalTier(state["tier"])
        nxt = ESCALATION_NEXT[current]  # guarded non-None by _after_gate
        escalations = state.get("escalations", 0) + 1
        entry = trace_entry(
            self.NODE_ESCALATE,
            perf_counter() - t0,
            detail={
                "from_tier": current.value,
                "to_tier": nxt.value,
                "escalations": escalations,
            },
        )
        return {
            "force_min_tier": nxt.value,
            "escalations": escalations,
            "trace": [entry],
        }

    # ------------------------------------------------------------------
    # Conditional edge
    # ------------------------------------------------------------------

    def _after_gate(self, state: RAGState) -> str:
        """Decide escalate vs END.

        Escalate only when the verdict is ``incorrect``, the per-query bound
        ``settings.max_escalations`` has not been reached, and a higher tier
        exists. At tier C, or once the bound is hit, the answer stands.
        """
        if (
            state.get("verdict") == "incorrect"
            and state.get("escalations", 0) < self.settings.max_escalations
            and ESCALATION_NEXT[RetrievalTier(state["tier"])] is not None
        ):
            return "escalate"
        return "end"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> "CompiledStateGraph":
        """Compile the LangGraph state machine (cached for :meth:`answer`)."""
        graph = StateGraph(RAGState)
        graph.add_node(self.NODE_ROUTE, self._route_node)
        graph.add_node(self.NODE_EXECUTE, self._execute_node)
        graph.add_node(self.NODE_GATE, self._gate_node)
        graph.add_node(self.NODE_ESCALATE, self._escalate_node)
        graph.add_edge(START, self.NODE_ROUTE)
        graph.add_edge(self.NODE_ROUTE, self.NODE_EXECUTE)
        graph.add_edge(self.NODE_EXECUTE, self.NODE_GATE)
        graph.add_conditional_edges(
            self.NODE_GATE,
            self._after_gate,
            {"escalate": self.NODE_ESCALATE, "end": END},
        )
        graph.add_edge(self.NODE_ESCALATE, self.NODE_ROUTE)  # the back-edge
        self._compiled = graph.compile()
        return self._compiled

    def answer(self, question: str) -> RAGState:
        """Answer one question; returns the final ``RAGState`` dict with
        ``answer``, final ``tier``, ``verdict``, ``escalations`` and the
        full per-node ``trace``."""
        app = self._compiled or self.build()
        initial: RAGState = {
            "question": question,
            "force_min_tier": None,
            "escalations": 0,
            "trace": [],
        }
        return app.invoke(initial)
