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
        self._vectors: np.ndarray | None = None

    # -- embedding model -------------------------------------------------
    @property
    def device(self) -> str:
        """Resolve the embedding device. "auto" prefers CUDA, falls back to CPU."""
        configured = getattr(self.cfg, "device", "auto")
        if configured != "auto":
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        """Load the sentence-transformer on first use.

        Device matters more here than anywhere else in the project: embedding
        throughput is the entire ingest bottleneck. Measured at ~45 chunks/s on
        a 6-core CPU, which puts a 25 x 500-page ingest at ~26 minutes. A CUDA
        device is one to two orders of magnitude faster and is the only way a
        corpus that size lands inside a couple of minutes.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.embedding_model, device=self.device)
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
    def embed(self, texts: list[str], batch_size: int | None = None,
              progress=None) -> np.ndarray:
        """Batched, CPU-safe embedding. Returns float32 of shape (n, dim)."""
        if not texts:
            return np.zeros((0, self.cfg.embedding_dim), dtype="float32")
        if progress:
            progress(0, len(texts), "embedding (single process)")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size or self.cfg.embed_batch_size,
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
        self._vectors = vectors
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
        # Persist vectors so replacing one document never re-embeds the rest.
        if self._vectors is not None:
            self.cfg.vectors_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cfg.vectors_path, self._vectors)
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
        self._vectors = (
            np.load(self.cfg.vectors_path).astype("float32")
            if self.cfg.vectors_path.exists() else None
        )
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


    # -- bulk ingest (Phase 7+) ------------------------------------------
    def _worker_count(self, n_texts: int) -> int:
        """How many processes to embed with.

        Pool startup costs a few seconds, so it is only worth paying above
        `bulk_worker_threshold`. Defaults to physical cores (capped at 8):
        transformer inference on CPU scales better across processes than
        across threads, because each process gets its own BLAS thread pool.
        """
        if n_texts < self.cfg.bulk_worker_threshold:
            return 1
        workers = self.cfg.embed_workers
        if workers > 0:
            return workers
        import os

        return max(1, min(8, (os.cpu_count() or 2) // 2))

    def embed_parallel(self, texts: list[str], progress=None) -> np.ndarray:
        """Embed across worker processes, falling back to single-process.

        Falls back rather than failing: the multi-process pool needs a spawn
        context and can be unavailable (frozen apps, some container setups),
        and an ingest that works slowly beats an ingest that raises.
        """
        if not texts:
            return np.zeros((0, self.cfg.embedding_dim), dtype="float32")

        workers = self._worker_count(len(texts))
        if workers <= 1:
            return self.embed(texts, batch_size=self.cfg.embed_batch_size, progress=progress)

        try:
            pool = self.model.start_multi_process_pool(target_devices=["cpu"] * workers)
        except Exception:                                  # noqa: BLE001
            return self.embed(texts, batch_size=self.cfg.embed_batch_size, progress=progress)

        try:
            if progress:
                progress(0, len(texts), f"embedding across {workers} processes")
            vectors = self.model.encode_multi_process(
                texts,
                pool,
                batch_size=self.cfg.embed_batch_size,
                normalize_embeddings=self.cfg.normalize_embeddings,
            )
            if progress:
                progress(len(texts), len(texts), "embedding complete")
            return np.asarray(vectors, dtype="float32")
        except Exception:                                  # noqa: BLE001
            return self.embed(texts, batch_size=self.cfg.embed_batch_size, progress=progress)
        finally:
            try:
                self.model.stop_multi_process_pool(pool)
            except Exception:                              # noqa: BLE001
                pass

    def chunk_bulk(self, text: str, doc_id: str) -> list[Chunk]:
        """Chunk with the coarser bulk profile.

        Fewer, larger chunks: a 500-page document yields ~1,070 instead of
        ~2,800. Embedding is linear in chunk count and is the whole bottleneck,
        so this is the single most effective lever available on CPU. It trades
        retrieval precision for throughput, which is why it is opt-in per
        request and never touches the evaluation corpus.
        """
        from dataclasses import replace as _replace

        coarse = _replace(
            self.cfg,
            chunk_size=self.cfg.bulk_chunk_size,
            chunk_overlap=self.cfg.bulk_chunk_overlap,
        )
        return Retriever(coarse).chunk(text, doc_id)

    def add_documents(
        self,
        docs: dict[str, str],
        bulk: bool = False,
        progress=None,
    ) -> dict[str, int]:
        """Append documents to the live index, creating one if none is loaded.

        Returns {doc_id: chunks_added}.

        Chunk ids stay stable because they derive from the document id and the
        chunk's position within that document, not a global counter, so
        appending never renumbers an existing chunk. Re-ingesting the same
        doc_id REPLACES its chunks rather than duplicating them.

        Replacement reuses the stored vectors of surviving chunks. Re-embedding
        them - which an earlier version did - costs O(corpus) per upload and at
        70,000 chunks takes over twenty minutes on CPU, so a one-line
        correction to a small document would re-embed the entire corpus.
        """
        import faiss

        added: dict[str, int] = {}
        new_chunks: list[Chunk] = []
        replacing: set[str] = set()

        chunker = self.chunk_bulk if bulk else self.chunk
        for doc_id, text in docs.items():
            chunks = chunker(text, doc_id)
            if not chunks:
                added[doc_id] = 0
                continue
            replacing.add(chunks[0].doc_id)
            new_chunks.extend(chunks)
            added[doc_id] = len(chunks)

        if not new_chunks:
            return added

        if progress:
            progress(0, len(new_chunks), f"embedding {len(new_chunks):,} chunks")
        vectors = self.embed_parallel([c.text for c in new_chunks], progress=progress)

        keep_mask = [c.doc_id not in replacing for c in self._chunks]
        survivors = [c for c, keep in zip(self._chunks, keep_mask) if keep]

        kept_vectors = None
        if survivors:
            stored = self._stored_vectors()
            if stored is not None and len(stored) == len(self._chunks):
                kept_vectors = stored[np.array(keep_mask, dtype=bool)]
            else:
                # No usable vector store (an index built by an older version).
                # Re-embed once, then persist so it never happens again.
                kept_vectors = self.embed_parallel([c.text for c in survivors])

        self._chunks = survivors + new_chunks
        self._vectors = (
            np.vstack([kept_vectors, vectors]) if kept_vectors is not None else vectors
        )
        self._index = faiss.IndexFlatIP(self._vectors.shape[1])
        self._index.add(self._vectors)
        self._by_id = {c.chunk_id: c for c in self._chunks}

        if progress:
            progress(len(new_chunks), len(new_chunks), "writing index")
        self.save_index()
        return added

    def _stored_vectors(self) -> np.ndarray | None:
        """Vectors for the current chunks, from memory or from disk."""
        if getattr(self, "_vectors", None) is not None:
            return self._vectors
        path = self.cfg.vectors_path
        if path.exists():
            try:
                return np.load(path).astype("float32")
            except (ValueError, OSError):
                return None
        return None

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
