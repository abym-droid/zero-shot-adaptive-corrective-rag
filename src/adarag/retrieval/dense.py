"""Dense bi-encoder retriever (sentence-transformers + FAISS).

Embeddings are L2-normalized so a FAISS ``IndexFlatIP`` computes cosine
similarity; exact flat search is deliberate - every corpus in scope fits in
RAM, so there is no ANN tuning to get wrong. The encoder is injectable (any
object with a compatible ``encode``) so tests can run offline with a fake
embedder, same idea as the LLM layer's FakeBackend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adarag.config import settings
from adarag.retrieval.base import Doc, Retriever, chunk_to_doc

_INDEX_NAME = "index.faiss"
_CHUNKS_NAME = "chunks.jsonl"
_META_NAME = "meta.json"
_FORMAT_VERSION = 1


def _load_st_encoder(model_name: str, device: str = "auto"):
    """Load a SentenceTransformer on the best available device (lazy import)."""
    from sentence_transformers import SentenceTransformer

    from adarag.device import get_device

    return SentenceTransformer(model_name, device=get_device(device))


class DenseRetriever(Retriever):
    """FAISS inner-product search over normalized bi-encoder embeddings.

    Build with `DenseRetriever.build(chunks)`, persist with `save(dir)`,
    restore with `DenseRetriever.load(dir)`. The heavyweight encoder is
    loaded lazily on first use (build-time encode or first query).
    """

    name = "dense"

    def __init__(
        self,
        index,
        chunks: list[dict],
        model_name: str,
        encoder=None,
        device: str = "auto",
    ):
        """Wrap a populated FAISS index; prefer `build()` / `load()`.

        ``model_name`` is stored in ``meta.json`` so `load` re-creates the
        same encoder. An injected ``encoder`` must expose
        ``encode(list[str], normalize_embeddings=True, convert_to_numpy=True)
        -> np.ndarray``; when ``None``, a SentenceTransformer is created
        lazily from ``model_name``.
        """
        self._index = index
        self._chunks = chunks
        self.model_name = model_name
        self._encoder = encoder
        self._device = device

    def __len__(self) -> int:
        return len(self._chunks)

    # -- encoding -----------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_st_encoder(self.model_name, self._device)
        return self._encoder

    @staticmethod
    def _encode(encoder, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts to a contiguous float32, L2-normalized matrix."""
        emb = encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(np.asarray(emb, dtype=np.float32))

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[dict],
        model_name: str = settings.embedder_model,
        encoder=None,
        device: str = "auto",
        batch_size: int = 64,
    ) -> "DenseRetriever":
        """Embed all chunk texts and index them with FAISS.

        ``encoder`` can be injected for offline tests; otherwise a
        SentenceTransformer is loaded from ``model_name`` (default
        `settings.embedder_model`, MiniLM-L6-v2).
        """
        if not chunks:
            raise ValueError("DenseRetriever.build: chunks list is empty")
        import faiss

        enc = encoder if encoder is not None else _load_st_encoder(model_name, device)
        emb = cls._encode(enc, [c["text"] for c in chunks], batch_size=batch_size)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(index, list(chunks), model_name, encoder=enc, device=device)

    # -- search -------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[Doc]:
        """Cosine-similarity top-``k`` over the chunk index, best first."""
        if not query.strip():
            return []
        k = min(k, len(self._chunks))
        q = self._encode(self._get_encoder(), [query])
        scores, idx = self._index.search(q, k)
        return [
            chunk_to_doc(self._chunks[i], float(s))
            for s, i in zip(scores[0], idx[0])
            if i >= 0
        ]

    # -- persistence --------------------------------------------------------

    def save(self, out_dir: str | Path) -> Path:
        """Write ``index.faiss`` + ``chunks.jsonl`` + ``meta.json``; returns
        the output directory (created if missing)."""
        import faiss

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out_dir / _INDEX_NAME))
        with (out_dir / _CHUNKS_NAME).open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        meta = {
            "format_version": _FORMAT_VERSION,
            "model_name": self.model_name,
            "dim": int(self._index.d),
            "count": len(self._chunks),
            "normalized": True,
            "metric": "inner_product",
        }
        (out_dir / _META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir

    @classmethod
    def load(
        cls, in_dir: str | Path, encoder=None, device: str = "auto"
    ) -> "DenseRetriever":
        """Restore a retriever previously written by `save`; when ``encoder``
        is ``None`` the encoder named in ``meta.json`` is loaded lazily on
        first query."""
        import faiss

        # faiss's OpenMP pool segfaults next to the MLX runtime on Apple
        # Silicon once the index is big enough to engage threaded search
        # kernels (hit on the 642k-chunk telecom index; the 5.7k scifact one
        # never triggered it). Single-threaded search costs ~0.01s per query
        # at this scale, so cap it unconditionally.
        faiss.omp_set_num_threads(1)

        in_dir = Path(in_dir)
        meta = json.loads((in_dir / _META_NAME).read_text(encoding="utf-8"))
        index = faiss.read_index(str(in_dir / _INDEX_NAME))
        chunks = [
            json.loads(line)
            for line in (in_dir / _CHUNKS_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(index, chunks, meta["model_name"], encoder=encoder, device=device)
