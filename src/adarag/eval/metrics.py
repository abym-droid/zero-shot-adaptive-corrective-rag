"""Answer- and routing-quality metrics for the pipeline.

Covers SQuAD-style EM/F1 (standard normalisation: lowercase, strip articles
and punctuation, collapse whitespace), MCQ option-letter accuracy with a
text-EM fallback when no letter can be extracted, routing accuracy against
silver labels, Recall@K, and aggregate() which folds prediction rows into the
summary.json numbers. Everything is pure stdlib so it imports anywhere with
no model or network dependencies.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "normalize_answer",
    "em",
    "f1",
    "extract_option_letter",
    "mcq_em",
    "mcq_accuracy",
    "routing_accuracy",
    "recall_at_k",
    "aggregate",
]

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Option letters supported for MCQ extraction (TeleQnA has up to 5 options;
# keep headroom up to J for other benchmarks).
_MCQ_LETTERS = "ABCDEFGHIJ"

# "B", "(B)", "B.", "B:" ... at the very start of the prediction.
_LEADING_LETTER_RE = re.compile(
    rf"^\s*\(?([{_MCQ_LETTERS}{_MCQ_LETTERS.lower()}])\)?\s*(?:[).:,\-]|$|\s)"
)
# "the answer is (B)", "answer: B", "option B", "choice B" anywhere in the text.
_ANSWER_IS_RE = re.compile(
    rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([{_MCQ_LETTERS}{_MCQ_LETTERS.lower()}])\)?\b",
    re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Normalize an answer string SQuAD-style for EM/F1 comparison.

    Lowercase, strip punctuation, drop English articles, collapse whitespace -
    kept identical to the SQuAD v1.1 eval script so our EM/F1 numbers stay
    comparable with the Adaptive-RAG / CRAG baselines.
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    """Whitespace tokens of the normalized answer string."""
    return normalize_answer(text).split()


def em(pred: str, golds: Sequence[str]) -> float:
    """Exact Match: 1.0 if the normalized prediction equals any normalized gold."""
    npred = normalize_answer(pred)
    return float(any(npred == normalize_answer(g) for g in golds))


def f1(pred: str, golds: Sequence[str]) -> float:
    """Max token-level F1 between the prediction and any gold answer.

    Standard SQuAD bag-of-tokens F1, maximized over the gold set. If either
    side is empty the score is 1.0 only when both are empty.
    """
    pred_toks = _tokens(pred)
    best = 0.0
    for gold in golds:
        gold_toks = _tokens(gold)
        if not pred_toks or not gold_toks:
            score = float(pred_toks == gold_toks)
        else:
            common = Counter(pred_toks) & Counter(gold_toks)
            overlap = sum(common.values())
            if overlap == 0:
                score = 0.0
            else:
                precision = overlap / len(pred_toks)
                recall = overlap / len(gold_toks)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def extract_option_letter(
    text: str, options: Mapping[str, str] | Sequence[str] | None = None
) -> str | None:
    """Extract the MCQ option letter a model's free-text answer refers to.

    Tries, in order: a leading letter pattern ("B", "(B)", "B."), an
    "answer is B" / "option B" phrase, then - if ``options`` is given - an
    option whose text appears verbatim (normalized) inside the prediction,
    longest match winning to avoid substring collisions like "LTE" inside
    "LTE-Advanced". ``options`` can be a letter->text mapping or a positional
    sequence. Returns the uppercase letter or None.
    """
    if not text:
        return None

    m = _LEADING_LETTER_RE.match(text) or _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).upper()

    if options:
        opt_map = _as_option_map(options)
        npred = normalize_answer(text)
        best_letter: str | None = None
        best_len = 0
        for letter, opt_text in opt_map.items():
            nopt = normalize_answer(opt_text)
            if nopt and nopt in npred and len(nopt) > best_len:
                best_letter, best_len = letter.upper(), len(nopt)
        if best_letter is not None:
            return best_letter

    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in _MCQ_LETTERS:
        return stripped.upper()
    return None


def _as_option_map(
    options: Mapping[str, str] | Sequence[str],
) -> dict[str, str]:
    """Coerce options (mapping or positional sequence) to a letter->text dict."""
    if isinstance(options, Mapping):
        return {str(k).strip().upper(): str(v) for k, v in options.items()}
    return {_MCQ_LETTERS[i]: str(v) for i, v in enumerate(options)}


def _gold_letter(golds: Sequence[str]) -> str | None:
    """Pick the single-letter gold (data/schema.py stores letter AND text)."""
    for g in golds:
        s = str(g).strip()
        if len(s) == 1 and s.upper() in _MCQ_LETTERS:
            return s.upper()
    return None


def mcq_em(
    pred: str,
    golds: Sequence[str],
    options: Mapping[str, str] | Sequence[str] | None = None,
) -> float:
    """Option-letter EM for one MCQ example.

    Compares the letter extracted from ``pred`` with the gold letter found in
    ``golds`` (loaders store both the letter and the option text). Falls back
    to plain text em() when either letter is unavailable, so free-text answers
    that spell out the correct option still score.
    """
    gold = _gold_letter(golds)
    pred_letter = extract_option_letter(pred, options)
    if gold is not None and pred_letter is not None:
        return float(pred_letter == gold)
    # No gold letter (some datasets store only the option text): resolve the
    # predicted letter to its option text and score that against the golds.
    if pred_letter is not None and options:
        pred_text = _as_option_map(options).get(pred_letter)
        if pred_text is not None:
            return em(pred_text, golds)
    return em(pred, golds)


def mcq_f1(
    pred: str,
    golds: Sequence[str],
    options: Mapping[str, str] | Sequence[str] | None = None,
) -> float:
    """Token F1 for one MCQ example, scoring the chosen option's text.

    A bare-letter answer ("B") has no token overlap with a text gold
    ("CONTRADICT"), so the letter is resolved to its option text first,
    mirroring mcq_em. Falls back to plain f1() when no letter can be
    extracted.
    """
    pred_letter = extract_option_letter(pred, options)
    if pred_letter is not None and options:
        pred_text = _as_option_map(options).get(pred_letter)
        if pred_text is not None:
            return f1(pred_text, golds)
    return f1(pred, golds)


def mcq_accuracy(
    preds: Sequence[str],
    golds: Sequence[Sequence[str]],
    options: Sequence[Mapping[str, str] | Sequence[str] | None] | None = None,
) -> float:
    """Mean option-letter EM over a batch of aligned MCQ examples (0.0 if empty)."""
    if len(preds) != len(golds) or (options is not None and len(options) != len(preds)):
        raise ValueError("preds, golds (and options) must be aligned")
    if not preds:
        return 0.0
    opts = options or [None] * len(preds)
    return sum(mcq_em(p, g, o) for p, g, o in zip(preds, golds, opts)) / len(preds)


def _tier_value(tier: Any) -> str:
    """Coerce a RetrievalTier enum or plain string to its string value."""
    return getattr(tier, "value", tier)


def routing_accuracy(
    preds: Mapping[str, Any] | Sequence[Any],
    silver: Mapping[str, Any] | Sequence[Any],
) -> float:
    """Agreement between router tier decisions and silver routing labels.

    Accepts either two qid-keyed mappings (joined on common qids) or two
    aligned sequences. ``preds`` should be the router's *initial* tiers,
    before any escalation. Tier values may be RetrievalTier members or plain
    strings - compared by value. Returns 0.0 when there is nothing to compare.
    """
    if isinstance(preds, Mapping) and isinstance(silver, Mapping):
        qids = sorted(set(preds) & set(silver))
        pairs = [(preds[q], silver[q]) for q in qids]
    else:
        preds_l, silver_l = list(preds), list(silver)  # type: ignore[arg-type]
        if len(preds_l) != len(silver_l):
            raise ValueError("preds and silver sequences must be aligned")
        pairs = list(zip(preds_l, silver_l))
    if not pairs:
        return 0.0
    return sum(_tier_value(p) == _tier_value(s) for p, s in pairs) / len(pairs)


def recall_at_k(
    retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int | None = None
) -> float:
    """Recall@K: fraction of gold passage ids found in the top-K retrieved.

    ``k=None`` uses the full retrieved list. Returns 0.0 when ``gold_ids`` is
    empty - such rows carry no retrieval signal, so skip them when averaging.
    """
    if not gold_ids:
        return 0.0
    top = set(retrieved_ids if k is None else retrieved_ids[:k])
    return sum(g in top for g in gold_ids) / len(gold_ids)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-example prediction rows into the summary.json aggregates.

    Consumes the row schema written by run_eval; missing fields are tolerated
    so partial rows still aggregate. Returns mean EM/F1 (plus MCQ accuracy
    over MCQ rows), escalation rate, tier/verdict distributions, latency and
    token cost per query, and the example count.
    """
    n = len(predictions)
    summary: dict[str, Any] = {"n_examples": n}
    if n == 0:
        return summary

    summary["em"] = _mean([float(r.get("em", 0.0)) for r in predictions])
    summary["f1"] = _mean([float(r.get("f1", 0.0)) for r in predictions])

    mcq_rows = [r for r in predictions if r.get("is_mcq")]
    if mcq_rows:
        summary["mcq_accuracy"] = _mean([float(r.get("em", 0.0)) for r in mcq_rows])
        summary["n_mcq"] = len(mcq_rows)

    summary["escalation_rate"] = _mean(
        [float(bool(r.get("escalated"))) for r in predictions]
    )
    summary["tier_initial_counts"] = dict(
        Counter(_tier_value(r.get("tier_initial")) for r in predictions)
    )
    summary["tier_final_counts"] = dict(
        Counter(_tier_value(r.get("tier_final")) for r in predictions)
    )
    summary["verdict_counts"] = dict(
        Counter(str(r.get("verdict")) for r in predictions)
    )
    summary["mean_latency_s"] = _mean(
        [float(r.get("latency_s", 0.0)) for r in predictions]
    )
    summary["mean_prompt_tokens"] = _mean(
        [float(r.get("prompt_tokens", 0)) for r in predictions]
    )
    summary["mean_completion_tokens"] = _mean(
        [float(r.get("completion_tokens", 0)) for r in predictions]
    )
    summary["total_tokens"] = int(
        sum(
            float(r.get("prompt_tokens", 0)) + float(r.get("completion_tokens", 0))
            for r in predictions
        )
    )
    return summary
