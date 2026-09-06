"""Build the demo index. Run at Docker build time, not on first request.

    python -m app.build_demo_index

A first visitor who waits while a corpus is embedded assumes the app is broken.
The demo corpus is small enough that building it during the image build costs
seconds and removes that wait entirely.

Deliberately built rather than committed: a prebuilt FAISS file in git would be
a binary artifact that drifts from the chunker that produced it, and it would
not be reproducible from a clean clone. This script is.
"""

from __future__ import annotations

import sys

from app.api import app_retrieval_config
from app.demo_corpus import load_demo_documents
from core.retriever import Retriever


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    docs = load_demo_documents()
    if not docs:
        print("no demo documents found in data/demo/ - nothing to build")
        raise SystemExit(0)

    retriever = Retriever(app_retrieval_config())
    added = retriever.add_documents(docs)

    print(f"demo index built: {retriever.size} chunks from {len(docs)} documents")
    for doc_id, count in sorted(added.items()):
        print(f"  {doc_id}: {count} chunks")
    print(f"written to {retriever.cfg.index_path}")


if __name__ == "__main__":
    main()
