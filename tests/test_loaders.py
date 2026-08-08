"""Tests for the unified QA data schema and loader (data/schema.py, data/loaders.py).

Every dataset is normalized on disk to one jsonl format of QAExample rows
so eval/run_eval stays dataset-agnostic. Pins the jsonl round-trip,
including MCQ metadata (options dict) and the two-form answers list
[letter, option text].
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

from adarag.data.loaders import load_dataset_file
from adarag.data.schema import QAExample

ROWS = [
    {
        "qid": "teleqna-0001",
        "question": "Which 3GPP release introduced 5G NR?",
        "answers": ["B", "Release 15"],
        "domain": "telecom",
        "dataset": "teleqna",
        "meta": {
            "options": {
                "A": "Release 14",
                "B": "Release 15",
                "C": "Release 16",
                "D": "Release 17",
            }
        },
    },
    {
        "qid": "squad-0042",
        "question": "What is the capital of France?",
        "answers": ["Paris"],
        "domain": "general_singlehop",
        "dataset": "squad",
        "meta": {},
    },
]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_qaexample_is_dataclass_with_spec_fields():
    ex = QAExample(**ROWS[1])
    assert is_dataclass(ex)
    assert asdict(ex) == ROWS[1]


def test_load_dataset_file_round_trip(tmp_path):
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, ROWS)

    examples = load_dataset_file(path)

    assert len(examples) == 2
    assert all(isinstance(ex, QAExample) for ex in examples)
    for ex, row in zip(examples, ROWS):
        assert ex.qid == row["qid"]
        assert ex.question == row["question"]
        assert ex.answers == row["answers"]
        assert ex.domain == row["domain"]
        assert ex.dataset == row["dataset"]
        assert ex.meta == row["meta"]


def test_mcq_metadata_survives_round_trip(tmp_path):
    path = tmp_path / "mcq.jsonl"
    _write_jsonl(path, [ROWS[0]])

    (ex,) = load_dataset_file(path)

    # answers carries both the option letter and the option text; options in meta
    assert ex.answers[0] == "B"
    assert "Release 15" in ex.answers
    assert ex.meta["options"]["B"] == "Release 15"


def test_load_preserves_row_order(tmp_path):
    path = tmp_path / "ordered.jsonl"
    _write_jsonl(path, list(reversed(ROWS)))

    examples = load_dataset_file(path)

    assert [ex.qid for ex in examples] == [r["qid"] for r in reversed(ROWS)]
