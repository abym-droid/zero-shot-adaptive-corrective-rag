"""Corpus chunking.

Fixed-size overlapping word windows - the deliberately simple baseline
policy, since the adaptivity being studied sits downstream of chunking and
chunking is held constant. Input is jsonl with ``{"doc_id", "title",
"text"}`` rows (as written by the prepare script); output chunks
are ``{"chunk_id", "doc_id", "title", "text"}`` dicts with
``chunk_id = f"{doc_id}::c{n}"``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    by: str = "words",
) -> list[str]:
    """Split text into fixed-size overlapping windows of words or chars.

    Returns an empty list for empty/whitespace-only input. The final window
    is never a strict suffix-subset of the previous one: iteration stops at
    the window that reaches the end of the text. Raises ``ValueError`` for
    bad sizes or an unknown ``by``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")
    if by not in ("words", "chars"):
        raise ValueError(f"by must be 'words' or 'chars', got {by!r}")

    if by == "words":
        units: list[str] = text.split()
        join = " ".join
    else:
        units = list(text)
        join = "".join
    if not units:
        return []

    step = chunk_size - overlap
    out: list[str] = []
    i = 0
    while i < len(units):
        out.append(join(units[i : i + chunk_size]))
        if i + chunk_size >= len(units):
            break
        i += step
    return out


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Yield one dict per non-blank line of a jsonl file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_corpus(
    corpus: str | Path | Iterable[dict],
    chunk_size: int = 300,
    overlap: int = 50,
    by: str = "words",
) -> list[dict]:
    """Chunk a whole corpus (jsonl path or an iterable of row dicts) into a
    flat list of chunk dicts in document order; ``chunk_id`` counts from 0
    within each document, and documents with empty text contribute no chunks.
    """
    if isinstance(corpus, (str, Path)):
        rows: Iterable[dict] = _iter_jsonl(corpus)
    else:
        rows = corpus

    chunks: list[dict] = []
    for row in rows:
        doc_id = str(row["doc_id"])
        title = str(row.get("title", ""))
        for n, piece in enumerate(
            chunk_text(row["text"], chunk_size=chunk_size, overlap=overlap, by=by)
        ):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::c{n}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": piece,
                }
            )
    return chunks
