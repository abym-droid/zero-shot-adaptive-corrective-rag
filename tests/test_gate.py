"""Tests for the CRAG-style corrective gate (gate.py, prompts/gate_prompts.py).

Three-way verdict (correct | ambiguous | incorrect); the escalation side
lives in test_escalation.py. Here we pin the gate's own contract: both
evaluation paths (retrieved docs for tiers B/C, draft answer for tier A),
the ``useful_doc_ids`` keep-list, and the fallback to "ambiguous" when
the JSON can't be parsed.
"""
from __future__ import annotations

import pytest

from adarag.gate import CorrectiveGate, GateVerdict
from adarag.llm.base import GenResult
from adarag.llm.fake import FakeBackend
from adarag.prompts.gate_prompts import GATE_ANSWER_PROMPT, GATE_DOC_PROMPT
from adarag.retrieval.base import Doc


def _docs() -> list[Doc]:
    return [
        Doc(doc_id="c1", text="Paris is the capital of France.", score=9.5, meta={}),
        Doc(doc_id="c2", text="The Amazon is a river.", score=1.2, meta={}),
    ]


def test_gate_prompts_exist():
    assert isinstance(GATE_DOC_PROMPT, str) and GATE_DOC_PROMPT.strip()
    assert isinstance(GATE_ANSWER_PROMPT, str) and GATE_ANSWER_PROMPT.strip()


@pytest.mark.parametrize("verdict", ["correct", "ambiguous", "incorrect"])
def test_evaluate_returns_each_verdict(verdict, gate_json):
    backend = FakeBackend([gate_json(verdict)])
    gate = CorrectiveGate(backend)
    result, gen = gate.evaluate("What is the capital of France?", _docs())
    assert isinstance(result, GateVerdict)
    assert result.verdict == verdict
    assert isinstance(gen, GenResult)


def test_evaluate_carries_useful_doc_ids(gate_json):
    # CRAG-lite refinement: gate can name the docs worth keeping.
    backend = FakeBackend([gate_json("ambiguous", useful_doc_ids=["c1"])])
    result, _ = CorrectiveGate(backend).evaluate("capital of France?", _docs())
    assert result.useful_doc_ids == ["c1"]


def test_useful_doc_ids_defaults_to_empty(gate_json):
    import json

    backend = FakeBackend([json.dumps({"verdict": "correct", "reason": "ok"})])
    result, _ = CorrectiveGate(backend).evaluate("q", _docs())
    assert result.useful_doc_ids == []


@pytest.mark.parametrize("verdict", ["correct", "ambiguous", "incorrect"])
def test_evaluate_answer_path_for_no_retrieval_tier(verdict, gate_json):
    # Tier A has no docs: gate judges the draft parametric answer instead.
    backend = FakeBackend([gate_json(verdict)])
    gate = CorrectiveGate(backend)
    result, gen = gate.evaluate_answer(
        "What is the capital of France?", "Paris"
    )
    assert isinstance(result, GateVerdict)
    assert result.verdict == verdict
    assert isinstance(gen, GenResult)


def test_evaluate_json_failure_falls_back_to_ambiguous():
    # unparseable JSON degrades to "ambiguous" rather than blocking the pipeline
    backend = FakeBackend(["not json, sorry"] * 6)
    result, _ = CorrectiveGate(backend).evaluate("q", _docs())
    assert result.verdict == "ambiguous"


def test_evaluate_answer_json_failure_falls_back_to_ambiguous():
    backend = FakeBackend(["still not json"] * 6)
    result, _ = CorrectiveGate(backend).evaluate_answer("q", "some answer")
    assert result.verdict == "ambiguous"
