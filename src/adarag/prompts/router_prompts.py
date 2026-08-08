"""Prompt variants for the zero-shot LLM complexity router.

``v1`` and ``v2`` share identical tier definitions and JSON instructions;
the only difference is that ``v2`` adds three few-shot examples (one per
tier), which keeps the prompt-sensitivity ablation a clean few-shot
comparison. Tier semantics deliberately mirror Adaptive-RAG (Jeong et al.,
2024). Templates contain literal JSON braces, so render them via
:func:`render_router_prompt` (plain replacement of ``{question}``), never
``str.format``. Once experiments start these strings are frozen - any edit
invalidates prompt-sensitivity results collected so far.
"""
from __future__ import annotations

__all__ = [
    "PROMPT_VARIANTS",
    "ROUTER_PROMPTS",
    "ROUTER_SYSTEM",
    "render_router_prompt",
]

ROUTER_SYSTEM: str = (
    "You are a query-complexity router inside a retrieval-augmented generation "
    "(RAG) system. Your only job is to pick the cheapest retrieval tier that "
    "can still answer the question correctly. You reply with a single JSON "
    "object and nothing else."
)
"""System message shared by both prompt variants."""

# Shared building blocks - keeping these identical across variants is what
# makes the v1-vs-v2 comparison a clean few-shot ablation.
_TIER_DEFINITIONS = """\
Tier definitions (choose exactly one):
- "no_retrieval" (Tier A): the question asks about common, stable, widely \
known facts that a strong language model already knows. Answering from \
parametric knowledge alone is sufficient; retrieving documents would add \
latency and cost but no new evidence.
- "single_step" (Tier B): the question is a single-hop factual question about \
a specific entity, quantity, date, definition, or detail that may be rare, \
recent, or domain-specific (for example from technical standards, medical or \
scientific literature). One retrieval pass over the document collection \
supplies the needed evidence.
- "iterative" (Tier C): the question is multi-hop or compositional. Answering \
requires chaining several distinct pieces of evidence - finding one fact and \
using it to look up the next - so repeated retrieve-and-reason steps are \
needed."""

_JSON_INSTRUCTION = """\
Respond with ONLY one JSON object, with no markdown fences and no text before \
or after it, in exactly this form:
{"tier": "no_retrieval" or "single_step" or "iterative", "reason": "one short sentence"}"""

_V1 = f"""\
Classify the question below into exactly one retrieval tier.

{_TIER_DEFINITIONS}

{_JSON_INSTRUCTION}

Question: {{question}}
JSON:"""

_FEW_SHOT_EXAMPLES = """\
Examples:

Question: What is the capital of Japan?
JSON: {"tier": "no_retrieval", "reason": "Common, stable world knowledge the model already holds; no evidence needed."}

Question: Who was the producer of the film The Faculty?
JSON: {"tier": "single_step", "reason": "Single-hop fact about one specific entity; one document lookup suffices."}

Question: Who is the spouse of the director of the film Jaws?
JSON: {"tier": "iterative", "reason": "Two-hop question: first find the director, then find that person's spouse."}"""

_V2 = f"""\
Classify the question below into exactly one retrieval tier.

{_TIER_DEFINITIONS}

{_JSON_INSTRUCTION}

{_FEW_SHOT_EXAMPLES}

Question: {{question}}
JSON:"""

ROUTER_PROMPTS: dict[str, str] = {"v1": _V1, "v2": _V2}
"""Registry of frozen router prompt templates keyed by variant name."""

PROMPT_VARIANTS: tuple[str, ...] = tuple(ROUTER_PROMPTS)
"""Valid variant names, in definition order: ``("v1", "v2")``."""


def render_router_prompt(variant: str, question: str) -> str:
    """Render a router prompt for one question.

    Plain string replacement of ``{question}`` rather than ``str.format``,
    so the literal JSON braces need no escaping and braces inside the user
    question cannot break rendering. Raises ``KeyError`` on an unknown
    variant.
    """
    if variant not in ROUTER_PROMPTS:
        raise KeyError(
            f"Unknown router prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}"
        )
    return ROUTER_PROMPTS[variant].replace("{question}", question.strip())
