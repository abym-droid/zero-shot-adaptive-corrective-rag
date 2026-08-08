"""CRAG-style corrective quality gate (Yan et al., 2024).

A gate LLM judges retrieved evidence (tiers B/C) or the parametric draft
answer (tier A) and returns ``correct`` / ``ambiguous`` / ``incorrect``.
The gate only issues the verdict - escalation on ``incorrect`` is the
orchestrator's job. Any LLM/JSON failure degrades to an ``ambiguous``
fallback so the gate never crashes the pipeline.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from adarag.prompts.gate_prompts import GATE_ANSWER_PROMPT, GATE_DOC_PROMPT

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime coupling
    from adarag.llm.base import GenResult, LLMBackend
    from adarag.retrieval.base import Doc

__all__ = ["GateVerdict", "CorrectiveGate"]


class GateVerdict(BaseModel):
    """Three-way judgment from the corrective gate. ``useful_doc_ids`` is the
    CRAG-lite keep-list of retrieved docs worth using as context (always
    empty on the tier-A path); ``reason`` is trace/debugging only.
    """

    verdict: Literal["correct", "ambiguous", "incorrect"]
    reason: str = ""
    useful_doc_ids: list[str] = Field(default_factory=list)


class CorrectiveGate:
    """CRAG-style evidence/answer evaluator backed by a zero-shot LLM.

    Parsing, retries and JSON extraction live in the backend
    (``generate_json`` with :class:`GateVerdict` as the schema); on any
    failure the gate returns the safe ``ambiguous`` fallback instead of
    raising. ``max_doc_chars`` caps each document rendered into the prompt
    to keep the judgment call cheap.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        max_doc_chars: int = 1500,
        max_tokens: int = 256,
    ) -> None:
        self.backend = backend
        self.max_doc_chars = max_doc_chars
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(self, question: str, docs: list[Doc]) -> tuple[GateVerdict, GenResult]:
        """Judge retrieved evidence quality for the question (tiers B/C).

        An empty doc list is a retrieval failure: deterministic ``incorrect``
        verdict with no LLM call, so the orchestrator can escalate. The
        returned ``useful_doc_ids`` are filtered to ids actually shown to
        the model.
        """
        if not docs:
            # Nothing retrieved: unambiguously bad evidence. Deterministic
            # verdict, zero cost - lets the graph escalate without an LLM call.
            return (
                GateVerdict(
                    verdict="incorrect",
                    reason="no documents retrieved; evidence unavailable",
                ),
                self._zero_gen_result(),
            )

        prompt = GATE_DOC_PROMPT.format(
            question=question, docs_block=self._format_docs(docs)
        )
        started = time.perf_counter()
        try:
            parsed, gen = self.backend.generate_json(
                prompt, GateVerdict, max_tokens=self.max_tokens
            )
        except Exception as exc:  # noqa: BLE001 - contract: never crash the pipeline
            return self._fallback(exc, started)

        verdict = GateVerdict.model_validate(parsed.model_dump())
        # CRAG-lite hygiene: drop hallucinated ids the model was never shown.
        known_ids = {doc.doc_id for doc in docs}
        verdict.useful_doc_ids = [i for i in verdict.useful_doc_ids if i in known_ids]
        return verdict, gen

    def evaluate_answer(self, question: str, answer: str) -> tuple[GateVerdict, GenResult]:
        """Judge a parametric draft answer's supportability (tier A path).

        No documents exist here, so the gate instead judges whether the
        closed-book draft looks confident and plausible; ``incorrect``
        signals the orchestrator to escalate A to B. An empty/whitespace
        draft yields a deterministic ``incorrect`` with no LLM call.
        """
        if not answer or not answer.strip():
            return (
                GateVerdict(
                    verdict="incorrect",
                    reason="empty draft answer from parametric generation",
                ),
                self._zero_gen_result(),
            )

        prompt = GATE_ANSWER_PROMPT.format(question=question, answer=answer.strip())
        started = time.perf_counter()
        try:
            parsed, gen = self.backend.generate_json(
                prompt, GateVerdict, max_tokens=self.max_tokens
            )
        except Exception as exc:  # noqa: BLE001 - contract: never crash the pipeline
            return self._fallback(exc, started)

        verdict = GateVerdict.model_validate(parsed.model_dump())
        verdict.useful_doc_ids = []  # no docs exist on the tier-A path
        return verdict, gen

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _format_docs(self, docs: list[Doc]) -> str:
        """Render docs as ``[doc_id] text`` lines, truncated per doc budget."""
        lines: list[str] = []
        for doc in docs:
            text = " ".join(doc.text.split())  # collapse whitespace/newlines
            if len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars].rstrip() + " …"
            lines.append(f"[{doc.doc_id}] {text}")
        return "\n".join(lines)

    def _fallback(self, exc: Exception, started: float) -> tuple[GateVerdict, GenResult]:
        """Build the safe ``ambiguous`` fallback after an LLM/JSON failure."""
        verdict = GateVerdict(
            verdict="ambiguous",
            reason=f"gate fallback ({type(exc).__name__}): could not obtain a "
            "valid JSON judgment; proceeding without corrective action",
        )
        return verdict, self._zero_gen_result(latency_s=time.perf_counter() - started)

    @staticmethod
    def _zero_gen_result(latency_s: float = 0.0) -> GenResult:
        """A zero-token ``GenResult`` stub for deterministic/fallback paths.

        Imported lazily so this module doesn't hard-depend on
        ``adarag.llm.base`` at import time.
        """
        from adarag.llm.base import GenResult

        return GenResult(
            text="", prompt_tokens=0, completion_tokens=0, latency_s=latency_s
        )
