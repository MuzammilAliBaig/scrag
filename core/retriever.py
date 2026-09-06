"""Module A - Ingest / Index.

docs -> vector index; query -> top-k chunks.

Chunk IDs are the load-bearing detail here. Module C cites them per sentence
and Module D resolves them back to text, so they are of the form
{doc_id}::{chunk_index} and must survive re-indexing. Never renumber silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import config
from core.types import Chunk

_WHITESPACE = re.compile(r"\s+")

CHUNK_ID_SEPARATOR = "::"


def normalize_doc_id(raw: str) -> str:
    """Make a doc_id safe to embed in a chunk id.

    The separator is a double colon, so a doc_id containing one would make ids
    ambiguous. Whitespace collapses to underscores.
    """
    cleaned = _WHITESPACE.sub("_", raw.strip())
    return cleaned.replace(CHUNK_ID_SEPARATOR, "__")


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Inverse of the id scheme. Raises ValueError on a malformed id."""
    doc_id, sep, index = chunk_id.rpartition(CHUNK_ID_SEPARATOR)
    if not sep or not index.isdigit():
        raise ValueError(f"malformed chunk id: {chunk_id!r}")
    return doc_id, int(index)


class Retriever:
    """Chunk, embed and store documents in FAISS; retrieve by similarity."""

    def __init__(self, cfg: config.RetrievalConfig = config.RETRIEVAL) -> None:
        self.cfg = cfg
        self._model = None          # loaded lazily; startup stays offline
        self._index = None
        self._chunks: list[Chunk] = []

    # -- embedding model -------------------------------------------------
    @property
    def model(self):
        """Load the sentence-transformer on first use, pinned to CPU."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.embedding_model, device="cpu")
        return self._model

    # -- chunking (1.1, 1.2) ---------------------------------------------
    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split one document into overlapping character windows.

        Windows are cut on a whitespace boundary where one exists near the end
        of the window, so chunks do not split words.
        """
        doc_id = normalize_doc_id(doc_id)
        text = text.strip()
        if not text:
            return []

        size, overlap = self.cfg.chunk_size, self.cfg.chunk_overlap
        if overlap >= size:
            raise ValueError(f"chunk_overlap ({overlap}) must be < chunk_size ({size})")

        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Prefer the last whitespace in the final quarter of the window.
                window_floor = end - size // 4
                pivot = text.rfind(" ", window_floor, end)
                if pivot > start:
                    end = pivot
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}{CHUNK_ID_SEPARATOR}{index}",
                        text=piece,
                        doc_id=doc_id,
                        start_char=start,
                        end_char=end,
                    )
                )
                index += 1
            if end >= len(text):
                break
            # Step back by the overlap, then snap to a word boundary. Without
            # the snap the next chunk opens mid-word ("162" instead of
            # "token162"), which corrupts the text Module C quotes and Module D
            # verifies. Snapping backwards keeps whole words and never shrinks
            # the overlap below what was configured.
            next_start = max(end - overlap, start + 1)
            if not text[next_start - 1].isspace():
                boundary = text.rfind(" ", start, next_start)
                if boundary > start:
                    next_start = boundary + 1
            start = next_start
        return chunks

    # -- embedding (1.3) -------------------------------------------------
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Batched, CPU-safe embedding. Returns float32 of shape (n, dim)."""
        if not texts:
            return np.zeros((0, self.cfg.embedding_dim), dtype="float32")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize_embeddings,
            show_progress_bar=len(texts) > 500,
        )
        return np.asarray(vectors, dtype="float32")

    # -- index build / save / load (1.4) ---------------------------------
    def build_index(self, docs: dict[str, str], cache_embeddings: bool = True) -> None:
        """Chunk, embed and index a mapping of doc_id to text, then persist it.

        Embeddings are cached alongside the index so a re-run does not re-embed
        an unchanged corpus.
        """
        import faiss

        self._chunks = [c for doc_id, text in docs.items() for c in self.chunk(text, doc_id)]
        if not self._chunks:
            raise ValueError("no chunks produced - corpus is empty")

        cache_path = config.DATASET.embedding_cache_path
        vectors = self._load_cached_embeddings(cache_path) if cache_embeddings else None
        if vectors is None:
            vectors = self.embed([c.text for c in self._chunks])
            if cache_embeddings:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, vectors)

        # Inner product on L2-normalized vectors is cosine similarity.
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self.save_index()

    def _load_cached_embeddings(self, cache_path: Path) -> np.ndarray | None:
        """Reuse cached vectors only if their count matches the current chunks."""
        if not cache_path.exists():
            return None
        vectors = np.load(cache_path)
        if vectors.shape[0] != len(self._chunks):
            return None  # corpus changed; re-embed rather than mismatch silently
        return vectors.astype("float32")

    def save_index(self, path: Path | None = None) -> None:
        import faiss

        path = path or self.cfg.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        meta = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "doc_id": c.doc_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in self._chunks
        ]
        self.cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.metadata_path.write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def load_index(self, path: Path | None = None) -> None:
        import faiss

        path = path or self.cfg.index_path
        if not path.exists():
            raise FileNotFoundError(f"no index at {path} - build it first")
        self._index = faiss.read_index(str(path))
        meta = json.loads(self.cfg.metadata_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**row) for row in meta]
        self._by_id = {c.chunk_id: c for c in self._chunks}
        if self._index.ntotal != len(self._chunks):
            raise ValueError(
                f"index/metadata mismatch: {self._index.ntotal} vectors, "
                f"{len(self._chunks)} chunks"
            )

    # -- retrieval (1.5) -------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the top-k chunks for a query, each carrying its score."""
        if self._index is None:
            raise RuntimeError("index not loaded - call build_index() or load_index()")
        k = min(k or self.cfg.top_k, self._index.ntotal)
        scores, indices = self._index.search(self.embed([query]), k)
        hits: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            base = self._chunks[idx]
            hits.append(
                Chunk(
                    chunk_id=base.chunk_id,
                    text=base.text,
                    doc_id=base.doc_id,
                    start_char=base.start_char,
                    end_char=base.end_char,
                    score=float(score),
                    metadata=base.metadata,
                )
            )
        return hits


    def add_documents(self, docs: dict[str, str]) -> dict[str, int]:
        """Append documents to the live index, creating one if none is loaded.

        Used by POST /ingest. Returns {doc_id: chunks_added}.

        Chunk ids stay stable because they are derived from the document id and
        the chunk's position within that document, not from a global counter -
        so appending never renumbers an existing chunk. Re-ingesting the same
        doc_id replaces its chunks rather than duplicating them.
        """
        import faiss

        added: dict[str, int] = {}
        new_chunks: list[Chunk] = []
        replacing: set[str] = set()

        for doc_id, text in docs.items():
            chunks = self.chunk(text, doc_id)
            if not chunks:
                added[doc_id] = 0
                continue
            replacing.add(chunks[0].doc_id)
            new_chunks.extend(chunks)
            added[doc_id] = len(chunks)

        if not new_chunks:
            return added

        # Re-ingesting a document replaces it. FAISS IndexFlatIP has no cheap
        # delete, so when anything is replaced the index is rebuilt from the
        # surviving chunks plus the new ones.
        survivors = [c for c in self._chunks if c.doc_id not in replacing]
        rebuild = len(survivors) != len(self._chunks) or self._index is None

        vectors = self.embed([c.text for c in new_chunks])
        if rebuild:
            keep_vectors = (
                self.embed([c.text for c in survivors]) if survivors else None
            )
            self._chunks = survivors + new_chunks
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            if keep_vectors is not None and len(keep_vectors):
                self._index.add(keep_vectors)
            self._index.add(vectors)
        else:
            self._chunks = self._chunks + new_chunks
            self._index.add(vectors)

        self._by_id = {c.chunk_id: c for c in self._chunks}
        self.save_index()
        return added

    def ensure_index(self) -> bool:
        """Load the index from disk if one exists. Returns whether it is ready."""
        if self._index is not None:
            return True
        try:
            self.load_index()
            return True
        except (FileNotFoundError, ValueError):
            return False

    @property
    def documents(self) -> dict[str, int]:
        """Chunks per document, for the /health and UI index summary."""
        counts: dict[str, int] = {}
        for chunk in self._chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return counts

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Resolve a cited id back to its text. Used by Modules C and D."""
        if not hasattr(self, "_by_id") or len(self._by_id) != len(self._chunks):
            self._by_id = {c.chunk_id: c for c in self._chunks}
        return self._by_id.get(chunk_id)

    @property
    def size(self) -> int:
        return len(self._chunks)
