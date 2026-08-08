"""Tests for the BM25 baseline retriever (retrieval/bm25.py, retrieval/base.py).

rank-bm25 is pure Python, so this is the one real retriever the suite can
exercise without models or network. Pins the Retriever/Doc contract,
ranking sanity on the tiny corpus, top-k, and save/load persistence.
"""
from __future__ import annotations

from adarag.retrieval.base import Doc, Retriever
from adarag.retrieval.bm25 import BM25Retriever


def test_bm25_is_a_retriever(bm25_retriever):
    assert issubclass(BM25Retriever, Retriever)
    assert isinstance(bm25_retriever.name, str) and bm25_retriever.name


def test_search_returns_docs_with_contract_fields(bm25_retriever):
    docs = bm25_retriever.search("capital of France", k=3)
    assert docs and len(docs) <= 3
    for d in docs:
        assert isinstance(d, Doc)
        assert isinstance(d.doc_id, str) and d.doc_id
        assert isinstance(d.text, str) and d.text
        assert isinstance(d.score, float)
        assert isinstance(d.meta, dict)


def test_search_ranks_topically_relevant_chunk_first(bm25_retriever):
    top = bm25_retriever.search("capital of France", k=1)[0]
    assert "capital of France" in top.text

    top = bm25_retriever.search("blood glucose hormone pancreas", k=1)[0]
    assert "Insulin" in top.text or "insulin" in top.text.lower()


def test_search_scores_descending(bm25_retriever):
    docs = bm25_retriever.search("river water", k=5)
    scores = [d.score for d in docs]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(bm25_retriever):
    assert len(bm25_retriever.search("Paris", k=2)) <= 2


def test_save_load_round_trip(bm25_retriever, tmp_path):
    bm25_retriever.save(tmp_path)
    reloaded = BM25Retriever.load(tmp_path)

    orig = bm25_retriever.search("Eiffel Tower Paris", k=2)
    back = reloaded.search("Eiffel Tower Paris", k=2)
    assert [d.doc_id for d in orig] == [d.doc_id for d in back]
