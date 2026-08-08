"""Tests for the zero-shot LLM router (router.py, prompts/router_prompts.py).

The router is a zero-shot LLM emitting structured JSON over three tiers
(no trained classifier). Covers: valid JSON → tier, force_min_tier
promotion (the gate's escalation hint), and garbage output → safe
SINGLE_STEP fallback.
"""
from __future__ import annotations

from adarag.config import RetrievalTier
from adarag.llm.base import GenResult
from adarag.llm.fake import FakeBackend
from adarag.prompts.router_prompts import ROUTER_PROMPTS
from adarag.router import RouteDecision, Router


def test_router_prompt_variants_exist_v1_and_v2():
    # the prompt-sensitivity ablation needs at least these two frozen variants
    assert {"v1", "v2"} <= set(ROUTER_PROMPTS)
    assert all(isinstance(p, str) and p.strip() for p in ROUTER_PROMPTS.values())


def test_route_valid_json_maps_to_tier(router_json):
    for tier in RetrievalTier:
        backend = FakeBackend([router_json(tier.value)])
        decision, gen = Router(backend).route("Some question?")
        assert isinstance(decision, RouteDecision)
        assert decision.tier == tier
        assert isinstance(gen, GenResult)


def test_route_with_v2_prompt_variant(router_json):
    backend = FakeBackend([router_json(RetrievalTier.ITERATIVE.value)])
    decision, _ = Router(backend, prompt_variant="v2").route(
        "Who directed the film whose lead actor was born in Lyon?"
    )
    assert decision.tier == RetrievalTier.ITERATIVE


def test_force_min_tier_promotes_lower_decision(router_json):
    # Router says A but gate escalation demands at least B → promoted to B.
    backend = FakeBackend([router_json(RetrievalTier.NO_RETRIEVAL.value)])
    decision, _ = Router(backend).route(
        "q", force_min_tier=RetrievalTier.SINGLE_STEP
    )
    assert decision.tier == RetrievalTier.SINGLE_STEP
    assert "promoted" in decision.reason.lower()


def test_force_min_tier_does_not_demote_higher_decision(router_json):
    # Router already chose C; a force_min_tier of B must not pull it down.
    backend = FakeBackend([router_json(RetrievalTier.ITERATIVE.value)])
    decision, _ = Router(backend).route(
        "q", force_min_tier=RetrievalTier.SINGLE_STEP
    )
    assert decision.tier == RetrievalTier.ITERATIVE


def test_force_min_tier_equal_is_noop(router_json):
    backend = FakeBackend([router_json(RetrievalTier.SINGLE_STEP.value)])
    decision, _ = Router(backend).route(
        "q", force_min_tier=RetrievalTier.SINGLE_STEP
    )
    assert decision.tier == RetrievalTier.SINGLE_STEP


def test_garbage_output_falls_back_to_single_step():
    # on LLMJsonError the router defaults to SINGLE_STEP with reason="fallback"
    backend = FakeBackend(["I refuse to emit JSON."] * 6)
    decision, _ = Router(backend).route("q")
    assert decision.tier == RetrievalTier.SINGLE_STEP
    assert decision.reason == "fallback"


def test_garbage_output_with_force_min_tier_still_respects_hint():
    # Fallback must not undercut an escalation hint above SINGLE_STEP.
    backend = FakeBackend(["nonsense"] * 6)
    decision, _ = Router(backend).route(
        "q", force_min_tier=RetrievalTier.ITERATIVE
    )
    assert decision.tier == RetrievalTier.ITERATIVE
