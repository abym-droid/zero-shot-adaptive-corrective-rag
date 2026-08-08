"""Zero-shot LLM query-complexity router.

Replaces Adaptive-RAG's trained T5 classifier (Jeong et al., 2024) with a
training-free LLM emitting a structured JSON decision over the same three
tiers (A no_retrieval / B single_step / C iterative). On escalation the
orchestrator re-enters the router with a ``force_min_tier`` hint, and the
router guarantees the decision is at least that tier. Prompt variants live
in ``adarag.prompts.router_prompts``.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, field_validator

from adarag.config import RetrievalTier
from adarag.llm.base import GenResult, LLMBackend, LLMJsonError
from adarag.prompts.router_prompts import (
    PROMPT_VARIANTS,
    ROUTER_SYSTEM,
    render_router_prompt,
)

__all__ = ["RouteDecision", "Router"]

logger = logging.getLogger(__name__)

# Tier ordering used for the force_min_tier promotion (A < B < C).
_TIER_ORDER: dict[RetrievalTier, int] = {
    RetrievalTier.NO_RETRIEVAL: 0,
    RetrievalTier.SINGLE_STEP: 1,
    RetrievalTier.ITERATIVE: 2,
}

# Lenient aliases for tier strings a smaller LLM might emit despite the
# JSON-only instruction. Keys are lower-cased before lookup.
_TIER_ALIASES: dict[str, str] = {
    "a": RetrievalTier.NO_RETRIEVAL.value,
    "tier a": RetrievalTier.NO_RETRIEVAL.value,
    "no-retrieval": RetrievalTier.NO_RETRIEVAL.value,
    "no retrieval": RetrievalTier.NO_RETRIEVAL.value,
    "none": RetrievalTier.NO_RETRIEVAL.value,
    "b": RetrievalTier.SINGLE_STEP.value,
    "tier b": RetrievalTier.SINGLE_STEP.value,
    "single-step": RetrievalTier.SINGLE_STEP.value,
    "single step": RetrievalTier.SINGLE_STEP.value,
    "single": RetrievalTier.SINGLE_STEP.value,
    "c": RetrievalTier.ITERATIVE.value,
    "tier c": RetrievalTier.ITERATIVE.value,
    "multi-step": RetrievalTier.ITERATIVE.value,
    "multi step": RetrievalTier.ITERATIVE.value,
    "multi_step": RetrievalTier.ITERATIVE.value,
    "multi-hop": RetrievalTier.ITERATIVE.value,
}


class RouteDecision(BaseModel):
    """Structured routing decision from the router LLM: chosen tier plus a
    short free-text reason. When the escalation back-edge promotes the
    decision, the promotion is recorded in ``reason`` so traces stay auditable.
    """

    tier: RetrievalTier
    reason: str = ""

    @field_validator("tier", mode="before")
    @classmethod
    def _coerce_tier(cls, value: object) -> object:
        """Map alias strings (e.g. ``"B"``) onto canonical tier values."""
        if isinstance(value, str):
            key = value.strip().lower()
            return _TIER_ALIASES.get(key, key)
        return value


class Router:
    """Zero-shot LLM complexity router with structured JSON output.

    Wraps an ``LLMBackend`` (typically the same open-weight 7B model as the
    generator) and classifies each question into one of the three
    Adaptive-RAG tiers via ``generate_json`` - no training or fine-tuning
    involved. ``prompt_variant`` picks one of the frozen prompts
    (``"v1"`` concise, ``"v2"`` few-shot - the ablation pair).
    """

    #: Token budget for the routing call - the decision JSON is tiny.
    max_tokens: int = 128

    def __init__(self, backend: LLMBackend, prompt_variant: str = "v1") -> None:
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant {prompt_variant!r}; "
                f"expected one of {PROMPT_VARIANTS}"
            )
        self.backend = backend
        self.prompt_variant = prompt_variant

    def route(
        self,
        question: str,
        *,
        force_min_tier: RetrievalTier | None = None,
    ) -> tuple[RouteDecision, GenResult]:
        """Route one question to a retrieval tier.

        Returns the (possibly promoted) decision plus the routing call's
        ``GenResult``. If the backend exhausts its JSON retries we fall back
        to tier B (the safe middle default) with a zeroed ``GenResult`` -
        token counts for the failed attempts aren't recoverable. If
        ``force_min_tier`` is set (the gate's back-edge hint) and the fresh
        decision is below it, the decision is promoted; the
        one-promotion-per-query bound is enforced by the orchestrator's
        escalation counter, not here.
        """
        prompt = render_router_prompt(self.prompt_variant, question)
        try:
            parsed, gen = self.backend.generate_json(
                prompt,
                RouteDecision,
                system=ROUTER_SYSTEM,
                max_tokens=self.max_tokens,
            )
            decision = RouteDecision.model_validate(parsed)
        except LLMJsonError as exc:
            logger.warning(
                "Router JSON fallback (variant=%s): %s", self.prompt_variant, exc
            )
            decision = RouteDecision(tier=RetrievalTier.SINGLE_STEP, reason="fallback")
            gen = GenResult(
                text="", prompt_tokens=0, completion_tokens=0, latency_s=0.0
            )

        if (
            force_min_tier is not None
            and _TIER_ORDER[decision.tier] < _TIER_ORDER[force_min_tier]
        ):
            decision = RouteDecision(
                tier=force_min_tier,
                reason=(
                    f"{decision.reason} [promoted=True: "
                    f"{decision.tier.value} -> {force_min_tier.value} "
                    f"(force_min_tier)]"
                ).strip(),
            )
        return decision, gen
