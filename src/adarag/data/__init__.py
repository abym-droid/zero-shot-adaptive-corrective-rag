"""Data layer: the unified QAExample schema plus per-dataset normalisers.

Every benchmark gets normalised into the same jsonl row shape so the rest of
the system never cares which dataset a question came from. Heavy deps like
``datasets`` are only imported inside the download script, so importing this
package stays cheap.
"""
from __future__ import annotations

from adarag.data.schema import (
    DOMAINS,
    QAExample,
    read_jsonl,
    write_jsonl,
)
from adarag.data.loaders import (
    load_dataset_file,
    write_corpus_jsonl,
)

__all__ = [
    "DOMAINS",
    "QAExample",
    "read_jsonl",
    "write_jsonl",
    "load_dataset_file",
    "write_corpus_jsonl",
]
