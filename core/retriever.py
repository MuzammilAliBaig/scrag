"""Module A - Ingest / Index.

docs -> vector index; query -> top-k chunks.
Done when a query returns top-k relevant chunks. Built in Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import config
from core.types import Chunk


class Retriever:
    """Chunk, embed and store documents in FAISS; retrieve by similarity."""

    def __init__(self, cfg: config.RetrievalConfig = config.RETRIEVAL) -> None:
        self.cfg = cfg

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split one document into overlapping chunks."""
        raise NotImplementedError("Module A - Phase 1")

    def build_index(self, docs: dict[str, str]) -> None:
        """Embed {doc_id: text} and persist a FAISS index to cfg.index_path."""
        raise NotImplementedError("Module A - Phase 1")

    def load_index(self, path: Path | None = None) -> None:
        """Load a previously built index from disk."""
        raise NotImplementedError("Module A - Phase 1")

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the top-k chunks for a query, scored."""
        raise NotImplementedError("Module A - Phase 1")
