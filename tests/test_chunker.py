"""Tests for corpus chunking (corpus/chunk.py).

Corpora get chunked with a sliding word window before indexing. Pins the
chunk_text size/overlap contract and the chunk dict schema
({"chunk_id", "doc_id", "title", "text"}) that corpus/build.py and the
retrievers consume.
"""
from __future__ import annotations

import json

from adarag.corpus.chunk import chunk_corpus, chunk_text


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_yields_single_chunk():
    text = _words(40)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].split() == text.split()


def test_chunk_sizes_respect_chunk_size():
    chunks = chunk_text(_words(250), chunk_size=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert len(chunks[0].split()) == 100


def test_chunk_overlap_between_consecutive_chunks():
    chunks = chunk_text(_words(250), chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    first, second = chunks[0].split(), chunks[1].split()
    # last `overlap` words of chunk i reappear at the head of chunk i+1
    assert first[-20:] == second[:20]


def test_chunks_cover_all_words_in_order():
    n = 250
    chunks = chunk_text(_words(n), chunk_size=100, overlap=20)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(_words(n).split())


def test_zero_overlap_partitions_text():
    chunks = chunk_text(_words(200), chunk_size=50, overlap=0)
    assert [w for c in chunks for w in c.split()] == _words(200).split()


def test_chunk_corpus_schema_and_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"doc_id": "d1", "title": "Doc One", "text": _words(120)},
        {"doc_id": "d2", "title": "Doc Two", "text": _words(30)},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    chunks = chunk_corpus(corpus, chunk_size=50, overlap=10)

    assert chunks, "chunk_corpus returned nothing"
    for c in chunks:
        assert {"chunk_id", "doc_id", "title", "text"} <= set(c)
    # chunk ids unique across the whole corpus
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    # every source doc contributes at least one chunk; titles preserved
    by_doc = {c["doc_id"] for c in chunks}
    assert by_doc == {"d1", "d2"}
    assert {c["title"] for c in chunks} == {"Doc One", "Doc Two"}
    # d1 (120 words @ size 50) must split into multiple chunks
    assert sum(1 for c in chunks if c["doc_id"] == "d1") >= 2
