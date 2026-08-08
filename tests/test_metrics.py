"""Tests for evaluation metrics (eval/metrics.py).

Covers EM and F1 (SQuAD-style normalization), MCQ option-letter accuracy,
routing accuracy against silver labels, and retrieval Recall@K. Expected
values are hand-computed so a regression in normalization or token
overlap is caught exactly.
"""
from __future__ import annotations

import pytest

from adarag.eval.metrics import (
    aggregate,
    em,
    f1,
    mcq_accuracy,
    mcq_em,
    normalize_answer,
    recall_at_k,
    routing_accuracy,
)

# ---------------------------------------------------------------------------
# normalize_answer - SQuAD-style: lowercase, strip articles & punctuation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("A  quick   brown fox.", "quick brown fox"),
        ("an apple", "apple"),
        ("Paris", "paris"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------


def test_em_exact_match_after_normalization():
    assert em("The Paris!", ["paris"]) == 1.0


def test_em_matches_any_gold():
    assert em("42", ["forty-two", "42"]) == 1.0


def test_em_no_match():
    assert em("London", ["Paris"]) == 0.0


# ---------------------------------------------------------------------------
# F1 (token overlap, best over golds)
# ---------------------------------------------------------------------------


def test_f1_perfect():
    assert f1("Barack Obama", ["barack obama"]) == pytest.approx(1.0)


def test_f1_partial_overlap():
    # pred={barack, obama}, gold={obama}: p=1/2, r=1 → f1 = 2/3
    assert f1("Barack Obama", ["Obama"]) == pytest.approx(2 / 3)


def test_f1_partial_overlap_three_tokens():
    # pred={new, york, city}, gold={york, city}: p=2/3, r=1 → f1 = 0.8
    assert f1("new york city", ["york city"]) == pytest.approx(0.8)


def test_f1_zero_overlap():
    assert f1("London", ["Paris"]) == pytest.approx(0.0)


def test_f1_takes_best_gold():
    assert f1("Paris", ["London", "Paris France"]) > 0.0


# ---------------------------------------------------------------------------
# MCQ option-letter accuracy (eval scores MCQ via option-letter EM;
# answers = [correct option letter, correct option text])
# ---------------------------------------------------------------------------


def test_mcq_em_bare_letter_correct():
    assert mcq_em("B", ["B", "optical fibre"]) == 1.0


def test_mcq_em_letter_extracted_from_sentence():
    assert mcq_em("The answer is (C).", ["C", "5G NR"]) == 1.0


def test_mcq_em_wrong_letter():
    assert mcq_em("A", ["B", "optical fibre"]) == 0.0


def test_mcq_em_case_insensitive():
    assert mcq_em("b", ["B", "optical fibre"]) == 1.0


def test_mcq_em_matches_option_text_via_options():
    options = {"A": "copper wire", "B": "optical fibre"}
    assert mcq_em("It uses optical fibre.", ["B", "optical fibre"], options) == 1.0


def test_mcq_accuracy_batch_mean():
    preds = ["B", "A"]
    golds = [["B", "optical fibre"], ["B", "optical fibre"]]
    assert mcq_accuracy(preds, golds) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Routing accuracy vs silver labels
# ---------------------------------------------------------------------------


def test_routing_accuracy_half():
    preds = ["single_step", "iterative"]
    silver = ["single_step", "single_step"]
    assert routing_accuracy(preds, silver) == pytest.approx(0.5)


def test_routing_accuracy_perfect():
    labels = ["no_retrieval", "single_step", "iterative"]
    assert routing_accuracy(labels, list(labels)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "c"], ["a", "x"]) == pytest.approx(0.5)


def test_recall_at_k_full():
    assert recall_at_k(["a", "b", "c"], ["c", "a"]) == pytest.approx(1.0)


def test_recall_at_k_none():
    assert recall_at_k(["a", "b"], ["z"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregate over prediction rows
# ---------------------------------------------------------------------------


def test_aggregate_means():
    predictions = [
        {"em": 1.0, "f1": 1.0, "latency_s": 0.2},
        {"em": 0.0, "f1": 0.5, "latency_s": 0.4},
    ]
    summary = aggregate(predictions)
    assert isinstance(summary, dict)
    assert summary["em"] == pytest.approx(0.5)
    assert summary["f1"] == pytest.approx(0.75)
