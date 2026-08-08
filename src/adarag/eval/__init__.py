"""Evaluation package: metrics, silver routing labels and the eval runner.

metrics has SQuAD-style EM/F1, MCQ option-letter accuracy, routing accuracy
and Recall@K; silver_labels derives routing labels from dataset provenance
(Adaptive-RAG style); run_eval drives the pipeline over a dataset and writes
predictions.jsonl + summary.json per run.
"""
from adarag.eval.metrics import (
    aggregate,
    em,
    extract_option_letter,
    f1,
    mcq_accuracy,
    mcq_em,
    normalize_answer,
    recall_at_k,
    routing_accuracy,
)
from adarag.eval.silver_labels import (
    MULTI_HOP_DATASETS,
    SINGLE_HOP_DATASETS,
    label_from_predictions,
    silver_label,
    silver_labels_for,
)

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
    "silver_label",
    "silver_labels_for",
    "label_from_predictions",
    "SINGLE_HOP_DATASETS",
    "MULTI_HOP_DATASETS",
]
