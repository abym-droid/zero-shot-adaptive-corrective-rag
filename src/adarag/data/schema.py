"""Unified QA-example schema shared by every benchmark in the evaluation set.

Each dataset (telecom, medical, scientific, general-domain calibration) gets
normalised into one QAExample per line of a jsonl file, so the router,
pipeline and evaluator never need dataset-specific code. The per-dataset
normalisers live in loaders.py; this module owns the row shape and the
jsonl read/write helpers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Canonical domain tags - loaders, the download script and the evaluator all
# key off these exact strings (per-domain breakdowns, silver routing labels).
DOMAINS = (
    "telecom",
    "medical",
    "scientific",
    "general_multihop",
    "general_singlehop",
)


@dataclass
class QAExample:
    """One question/answer row in the unified benchmark format.

    qids are dataset-prefixed (e.g. "teleqna-0") to stay globally unique.
    For MCQ sets ``answers`` holds BOTH the gold option letter and the option
    text (e.g. ["B", "Denver Broncos"]) so the evaluator can score either
    letter accuracy or free-text EM/F1; free-text sets store the gold answer
    plus any aliases. Common ``meta`` keys: ``options`` (letter->text map for
    MCQ), ``context`` (gold passage when the dataset ships one),
    ``gold_doc_ids`` (for Recall@K), and an ``mcq`` bool flag.
    """

    qid: str
    question: str
    answers: list[str]
    domain: str
    dataset: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(
                f"unknown domain {self.domain!r}; expected one of {DOMAINS}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serialisable dict for one row."""
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "QAExample":
        """Rebuild a QAExample from a parsed jsonl row (meta defaults to {})."""
        return cls(
            qid=row["qid"],
            question=row["question"],
            answers=list(row["answers"]),
            domain=row["domain"],
            dataset=row["dataset"],
            meta=dict(row.get("meta", {})),
        )


def write_jsonl(examples: Iterable[QAExample], path: str | Path) -> int:
    """Write examples to a jsonl file (creating parent dirs); returns row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[QAExample]:
    """Yield one QAExample per non-empty line of a jsonl file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield QAExample.from_dict(json.loads(line))
