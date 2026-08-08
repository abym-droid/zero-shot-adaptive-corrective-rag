"""Dataset-provenance silver routing labels, following Adaptive-RAG.

The protocol from Jeong et al. (2024): single-hop datasets get labelled
SINGLE_STEP (tier B), multi-hop datasets ITERATIVE (tier C), and the optional
upgrade-to-A rule relabels an example NO_RETRIEVAL when a closed-book run
already answered it correctly (read lazily from an existing predictions.jsonl
via label_from_predictions, so no extra generation pass is needed). These are
silver labels, not gold - dataset provenance is imperfect, but it is the same
reference Adaptive-RAG reports against, which keeps our routing-accuracy
numbers directly comparable to the reproduced baseline.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from adarag.config import RetrievalTier

__all__ = [
    "SINGLE_HOP_DATASETS",
    "MULTI_HOP_DATASETS",
    "silver_label",
    "silver_labels_for",
    "label_from_predictions",
]

#: Datasets whose questions are answerable from a single retrieved passage
#: (general single-hop calibration sets + the domain QA sets).
SINGLE_HOP_DATASETS: frozenset[str] = frozenset(
    {
        "squad",
        "triviaqa",
        "popqa",
        "teleqna",
        "medqa",
        "pubmedqa",
        "scifact",
        "mirage",
    }
)

#: Datasets constructed to require multi-hop reasoning across passages.
MULTI_HOP_DATASETS: frozenset[str] = frozenset(
    {
        "hotpotqa",
        "musique",
        "2wikimultihopqa",
    }
)

# Domain strings (data/schema.py QAExample.domain) usable as a fallback when
# the dataset name is unknown.
_MULTI_HOP_DOMAINS = frozenset({"general_multihop"})


def silver_label(dataset: str, domain: str | None = None) -> RetrievalTier:
    """Silver routing label for one example, from dataset provenance.

    Dataset names are case-insensitive and tolerate a ``.dev500`` suffix or
    file extension; ``domain`` is only consulted when the name is not
    recognized. Unknown provenance defaults to SINGLE_STEP - the safe
    single-hop default, mirroring the router's own fallback tier.
    """
    name = dataset.lower().strip()
    # tolerate "hotpotqa.dev500" / "hotpotqa.jsonl" style names
    name = name.split(".", 1)[0]
    if name in MULTI_HOP_DATASETS:
        return RetrievalTier.ITERATIVE
    if name in SINGLE_HOP_DATASETS:
        return RetrievalTier.SINGLE_STEP
    if domain is not None and domain.lower() in _MULTI_HOP_DOMAINS:
        return RetrievalTier.ITERATIVE
    return RetrievalTier.SINGLE_STEP


def silver_labels_for(examples: Iterable[Any]) -> dict[str, RetrievalTier]:
    """Silver labels for a batch of QAExample-like objects, keyed by qid."""
    labels: dict[str, RetrievalTier] = {}
    for ex in examples:
        qid = _field(ex, "qid")
        dataset = _field(ex, "dataset", "") or ""
        domain = _field(ex, "domain", None)
        labels[str(qid)] = silver_label(str(dataset), domain)
    return labels


def label_from_predictions(
    predictions_file: str | Path,
    dataset: str | None = None,
    base: Mapping[str, RetrievalTier] | None = None,
) -> dict[str, RetrievalTier]:
    """Silver labels with the upgrade-to-A rule applied.

    Reads a predictions.jsonl from a closed-book (no-retrieval) run and
    upgrades every example the parametric model answered correctly (em == 1)
    to NO_RETRIEVAL; everything else keeps its provenance label. ``dataset``
    fills in when rows lack their own dataset field; ``base`` (precomputed
    provenance labels) overrides per-row derivation.
    """
    path = Path(predictions_file)
    labels: dict[str, RetrievalTier] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid"))
            if float(row.get("em", 0.0)) >= 1.0:
                labels[qid] = RetrievalTier.NO_RETRIEVAL
                continue
            if base is not None and qid in base:
                labels[qid] = base[qid]
            else:
                labels[qid] = silver_label(
                    str(row.get("dataset", dataset or "")), row.get("domain")
                )
    return labels


def _field(obj: Any, name: str, default: Any = ...) -> Any:
    """Read attribute or dict key ``name`` from a QAExample-like object."""
    if isinstance(obj, Mapping):
        if default is ...:
            return obj[name]
        return obj.get(name, default)
    if default is ...:
        return getattr(obj, name)
    return getattr(obj, name, default)
