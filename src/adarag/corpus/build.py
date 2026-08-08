"""Index building for corpus jsonl files.

`build_indices` is the single entry point that turns a corpus jsonl
(``{"doc_id","title","text"}`` rows) into the on-disk layout the pipeline
and eval runner consume:

    indices/<name>/
        bm25/           bm25.pkl                     (BM25Retriever.save)
        dense/          index.faiss, chunks.jsonl,
                        meta.json                    (DenseRetriever.save)
        meta.json       corpus provenance + chunking parameters

Invoked by the ``adarag index`` CLI subcommand; also runnable standalone:
``conda run -n adarag python -m adarag.corpus.build <corpus.jsonl> <out_dir>``.

ColBERT indices are intentionally NOT built here: ragatouille is an optional
extra with its own on-disk index workflow (see `adarag.retrieval.colbert`).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from adarag.config import settings
from adarag.corpus.chunk import chunk_corpus

_META_NAME = "meta.json"
_SUPPORTED_KINDS = ("bm25", "dense")


def build_indices(
    corpus_jsonl: str | Path,
    out_dir: str | Path,
    embedder: str | object = settings.embedder_model,
    kinds: Sequence[str] = ("bm25", "dense"),
    chunk_size: int = 300,
    overlap: int = 50,
    device: str = "auto",
) -> dict:
    """Chunk a corpus and build the requested index kinds on disk.

    ``embedder`` is a sentence-transformers model id, or an already-built
    encoder object (offline tests; must expose ``encode(...)`` per
    `DenseRetriever`). Chunk sizes/overlap are in words. Returns the
    metadata dict also written to ``<out_dir>/meta.json``; raises on unknown
    ``kinds``, a missing corpus file, or an empty corpus.
    """
    unknown = set(kinds) - set(_SUPPORTED_KINDS)
    if unknown:
        raise ValueError(
            f"Unsupported index kinds {sorted(unknown)}; supported: {_SUPPORTED_KINDS} "
            "(colbert indices are built separately, see adarag.retrieval.colbert)"
        )
    corpus_jsonl = Path(corpus_jsonl)
    if not corpus_jsonl.is_file():
        raise FileNotFoundError(f"corpus jsonl not found: {corpus_jsonl}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_corpus(corpus_jsonl, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"corpus {corpus_jsonl} produced zero chunks")
    n_docs = len({c["doc_id"] for c in chunks})

    embedder_is_object = not isinstance(embedder, str)
    built: list[str] = []
    if "bm25" in kinds:
        from adarag.retrieval.bm25 import BM25Retriever

        BM25Retriever.build(chunks).save(out_dir / "bm25")
        built.append("bm25")
    if "dense" in kinds:
        from adarag.retrieval.dense import DenseRetriever

        if embedder_is_object:
            retriever = DenseRetriever.build(
                chunks, model_name="injected-encoder", encoder=embedder, device=device
            )
        else:
            retriever = DenseRetriever.build(chunks, model_name=embedder, device=device)
        retriever.save(out_dir / "dense")
        built.append("dense")

    meta = {
        "corpus_jsonl": str(corpus_jsonl),
        "n_docs": n_docs,
        "n_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_by": "words",
        "kinds": built,
        "embedder": "injected-encoder" if embedder_is_object else embedder,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / _META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _main() -> None:  # pragma: no cover - thin CLI wrapper
    """Standalone typer entry (``python -m adarag.corpus.build``)."""
    import typer

    def cmd(
        corpus_jsonl: Path = typer.Argument(..., help="Corpus jsonl (doc_id/title/text rows)."),
        out_dir: Path = typer.Argument(..., help="Index root, e.g. indices/scifact."),
        embedder: str = typer.Option(settings.embedder_model, help="Dense encoder model id."),
        kinds: str = typer.Option("bm25,dense", help="Comma list: bm25,dense."),
        chunk_size: int = typer.Option(300, help="Chunk size in words."),
        overlap: int = typer.Option(50, help="Chunk overlap in words."),
    ) -> None:
        meta = build_indices(
            corpus_jsonl,
            out_dir,
            embedder=embedder,
            kinds=tuple(k.strip() for k in kinds.split(",") if k.strip()),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        typer.echo(json.dumps(meta, indent=2))

    typer.run(cmd)


if __name__ == "__main__":  # pragma: no cover
    _main()
