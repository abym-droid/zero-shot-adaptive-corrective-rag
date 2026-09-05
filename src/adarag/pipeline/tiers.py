"""Tier execution strategies for the adaptive RAG pipeline.

The three tiers the router chooses between: A answers from parametric
knowledge only, B does one retrieval pass, C runs an IRCoT-lite interleaved
retrieve-and-reason loop (Trivedi et al., 2023). All answer prompts demand
short factoid answers so downstream EM/F1 scoring stays meaningful; each
runner returns a :class:`TierResult` with the answer, evidence docs and
per-LLM-call telemetry for the run trace.
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:  # typing only - no runtime coupling
    from adarag.llm.base import GenResult, LLMBackend
    from adarag.retrieval.base import Doc, Retriever

# ---------------------------------------------------------------------------
# Prompts (tier prompts are pipeline-owned; router/gate prompts live in
# adarag.prompts).
# ---------------------------------------------------------------------------

QA_SYSTEM = (
    "You are a precise question-answering assistant. Answer with only the "
    "short factoid answer (a few words at most): no full sentences, no "
    "explanations, no punctuation beyond what the answer itself requires."
)

NO_RETRIEVAL_PROMPT = """Answer the question from your own knowledge.
Give only the short answer.

Question: {question}
Answer:"""

SINGLE_STEP_PROMPT = """Use the context passages to answer the question.
If the context does not contain the answer, answer from your own knowledge.
Give only the short answer.

Context:
{context}

Question: {question}
Answer:"""

IRCOT_STEP_PROMPT = """You are answering a multi-step question by interleaving \
retrieval and reasoning. Read the context passages and the reasoning so far, \
then write EXACTLY ONE of the following:
- the single next reasoning step (one sentence naming the fact still needed), or
- if the context already suffices, the line: ANSWER: <short answer>

Context:
{context}

Question: {question}

Reasoning so far:
{thoughts}

Next step:"""

FINAL_ANSWER_PROMPT = """Use the context passages and the reasoning steps to \
answer the question. Give only the short answer, prefixed with "ANSWER:".

Context:
{context}

Question: {question}

Reasoning steps:
{thoughts}

ANSWER:"""

_ANSWER_MARKER = "ANSWER:"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TierResult:
    """Outcome of executing one retrieval tier: the short answer, the
    evidence docs used (empty for tier A; deduplicated, first-retrieved
    order for tier C), and per-LLM-call telemetry dicts for the trace.
    """

    answer: str
    docs: list["Doc"] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(detail: str, res: "GenResult") -> dict:
    """Convert one ``GenResult`` into a trace step dict."""
    return {
        "detail": detail,
        "latency_s": round(res.latency_s, 4),
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
    }


def _doc_key(doc: "Doc") -> str:
    """Deduplication key for a doc: prefer the chunk id, fall back to doc_id."""
    meta: Mapping = getattr(doc, "meta", None) or {}
    return str(meta.get("chunk_id") or doc.doc_id)


def _format_context(docs: Iterable["Doc"]) -> str:
    """Render docs as numbered context passages (id + optional title + text)."""
    blocks: list[str] = []
    for i, d in enumerate(docs, start=1):
        title = (getattr(d, "meta", None) or {}).get("title", "")
        header = f"[{i}] (id={d.doc_id})" + (f" {title}" if title else "")
        blocks.append(f"{header}\n{d.text}")
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def extract_answer(text: str) -> str:
    """Extract the short answer: everything after the first ``ANSWER:``
    marker (first line only) if present, otherwise the stripped text itself.
    """
    if _ANSWER_MARKER in text:
        text = text.split(_ANSWER_MARKER, 1)[1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


# ---------------------------------------------------------------------------
# Tier runners
# ---------------------------------------------------------------------------


def run_no_retrieval(state: Mapping, backend: "LLMBackend") -> TierResult:
    """Tier A: answer from parametric knowledge, no retrieval."""
    question = state["question"]
    res = backend.generate(
        NO_RETRIEVAL_PROMPT.format(question=question),
        system=QA_SYSTEM,
        max_tokens=128,
        temperature=0.0,
    )
    return TierResult(answer=extract_answer(res.text), docs=[], steps=[_step("generate", res)])


def run_single_step(
    state: Mapping,
    backend: "LLMBackend",
    retriever: "Retriever",
    top_k: int,
) -> TierResult:
    """Tier B: one retrieval pass, then answer with the retrieved context."""
    question = state["question"]
    docs = retriever.search(question, k=top_k)
    res = backend.generate(
        SINGLE_STEP_PROMPT.format(context=_format_context(docs), question=question),
        system=QA_SYSTEM,
        max_tokens=128,
        temperature=0.0,
    )
    return TierResult(
        answer=extract_answer(res.text),
        docs=list(docs),
        steps=[_step("retrieve+generate", res)],
    )


def run_iterative(
    state: Mapping,
    backend: "LLMBackend",
    retriever: "Retriever",
    top_k: int,
    max_iters: int = 3,
) -> TierResult:
    """Tier C: IRCoT-lite interleaved retrieval and reasoning.

    Loop (at most ``max_iters`` times, kept small to bound cost): retrieve
    with the current query (initially the question), accumulate context
    deduplicated by chunk id, generate one reasoning step; stop when the
    step contains ``ANSWER:``, otherwise the step becomes the next retrieval
    query. If the budget runs out without an answer, a final extraction call
    runs over all accumulated context and reasoning.
    """
    question = state["question"]
    query = question
    collected: dict[str, "Doc"] = {}  # insertion-ordered dedup by chunk id
    thoughts: list[str] = []
    steps: list[dict] = []
    answer: str | None = None

    # Post-freeze ablation switch (default off): force at least this many
    # retrieval rounds before an ``ANSWER:`` is accepted. Set
    # ADARAG_TIERC_MIN_ITERS=2 to make the tier iterate at least twice.
    min_iters = int(os.environ.get("ADARAG_TIERC_MIN_ITERS", "1") or 1)

    for i in range(1, max_iters + 1):
        for doc in retriever.search(query, k=top_k):
            collected.setdefault(_doc_key(doc), doc)
        context = _format_context(collected.values())
        thoughts_text = (
            "\n".join(f"{j}. {t}" for j, t in enumerate(thoughts, start=1))
            if thoughts
            else "(none yet)"
        )
        res = backend.generate(
            IRCOT_STEP_PROMPT.format(
                context=context, question=question, thoughts=thoughts_text
            ),
            system=QA_SYSTEM,
            max_tokens=192,
            temperature=0.0,
        )
        step_text = res.text.strip()
        steps.append(_step(f"ircot_step_{i}", res))
        if _ANSWER_MARKER in step_text and i >= min_iters:
            answer = extract_answer(step_text)
            break
        if _ANSWER_MARKER in step_text:
            # Early answer suppressed by the minimum-rounds switch: keep the
            # reasoning as a thought and retrieve again with it.
            step_text = step_text.split(_ANSWER_MARKER)[0].strip() or step_text
        thoughts.append(step_text)
        query = step_text  # next retrieval query = the reasoning step

    if answer is None:  # budget exhausted - final answer extraction pass
        thoughts_text = "\n".join(
            f"{j}. {t}" for j, t in enumerate(thoughts, start=1)
        ) or "(none)"
        res = backend.generate(
            FINAL_ANSWER_PROMPT.format(
                context=_format_context(collected.values()),
                question=question,
                thoughts=thoughts_text,
            ),
            system=QA_SYSTEM,
            max_tokens=128,
            temperature=0.0,
        )
        steps.append(_step("final_answer", res))
        answer = extract_answer(res.text)

    return TierResult(answer=answer, docs=list(collected.values()), steps=steps)
