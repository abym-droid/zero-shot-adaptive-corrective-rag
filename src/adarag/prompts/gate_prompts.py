"""Prompt templates for the CRAG-style corrective quality gate.

Both templates elicit a three-way verdict (correct / ambiguous / incorrect,
following Yan et al., 2024) as a JSON-only reply matching
``adarag.gate.GateVerdict`` - the judgment only, never the escalation
action. They are ``str.format`` strings with literal JSON braces escaped as
``{{ }}``; placeholders are ``{question}``/``{docs_block}`` for
``GATE_DOC_PROMPT`` and ``{question}``/``{answer}`` for
``GATE_ANSWER_PROMPT``.
"""
from __future__ import annotations

#: Shared JSON output contract appended to both gate prompts. Kept as a single
#: constant so the two prompts cannot drift apart on the response schema.
GATE_JSON_INSTRUCTIONS = """\
Respond with ONLY a single JSON object and nothing else (no code fences, no
markdown, no commentary), exactly in this shape:
{{"verdict": "correct" | "ambiguous" | "incorrect", "reason": "<one short sentence>", "useful_doc_ids": [<zero or more document id strings>]}}"""


#: Evidence-quality judgment for the retrieval tiers (B: single-step,
#: C: iterative). CRAG-style: the gate judges whether the retrieved documents
#: contain enough evidence to answer the question, and optionally names the
#: documents worth keeping (CRAG-lite knowledge refinement).
GATE_DOC_PROMPT = (
    """You are a strict evidence-quality evaluator inside a retrieval-augmented
question-answering system. Your job is to judge whether the retrieved
documents below provide sufficient evidence to answer the question. You are
NOT answering the question yourself.

Question:
{question}

Retrieved documents (each line starts with its id in square brackets):
{docs_block}

Verdict definitions:
- "correct": at least one document contains evidence that directly supports
  answering the question.
- "ambiguous": the documents are partially relevant or the evidence is
  incomplete or unclear; an answer attempt is still reasonable.
- "incorrect": the documents are irrelevant or misleading for this question;
  better retrieval is needed.

In "useful_doc_ids", list ONLY the ids of documents genuinely useful for
answering the question (an empty list if none). Copy ids exactly as shown in
the square brackets. Keep "reason" to one short sentence.

"""
    + GATE_JSON_INSTRUCTIONS
)


#: Draft-answer supportability judgment for the no-retrieval tier (A). There
#: are no documents on this path, so the gate instead judges whether the
#: parametric (closed-book) draft answer looks confident and plausible, or
#: whether the query should be escalated to retrieval (A -> B).
GATE_ANSWER_PROMPT = (
    """You are a strict answer-quality evaluator inside a question-answering
system. The draft answer below was produced from the model's parametric
knowledge alone, WITHOUT retrieving any documents. Judge whether the draft
answer is a confident, plausible, non-evasive answer to the question. You are
NOT answering the question yourself.

Question:
{question}

Draft answer:
{answer}

Verdict definitions:
- "correct": the draft directly answers the question and looks factually
  plausible and self-consistent.
- "ambiguous": the draft is partially responsive, hedged, or of uncertain
  correctness.
- "incorrect": the draft is evasive, self-contradictory, clearly wrong, or
  admits it does not know; the question likely needs document retrieval.

There are no documents on this path, so "useful_doc_ids" MUST be an empty
list. Keep "reason" to one short sentence.

"""
    + GATE_JSON_INSTRUCTIONS
)
