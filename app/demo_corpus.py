"""Demo corpus loader, so the app is not empty on first open.

Four short policy documents for a fictional institution. They are deliberately
narrow: the point of the demo is not that the system answers questions, but
that it **refuses** when the documents do not cover something, and shows you
which passage supports each sentence when they do.

**A caveat found by running this.** Module B was fine-tuned on PopQA entity
questions over Wikipedia lead sections. This policy corpus is out of
distribution for it, and measured on these four documents it grades almost
every chunk `ambiguous` - it even returns `correct` for "Who is the
Vice-Chancellor?", which the documents never mention. So the free
pre-generation refusal (trigger b) does not fire reliably here, and abstention
falls to Modules C, D and E. Those still work; the guarantee holds, it just
costs a generation call instead of being free.

The suggested questions below are split accordingly. The unanswerable ones are
not trick questions - they are ordinary things a student would ask that these
four documents simply do not address, which is exactly when a RAG system
normally invents an answer.
"""

from __future__ import annotations

from pathlib import Path

import config

DEMO_DIR = config.REPO_ROOT / "data" / "demo"

ANSWERABLE_QUESTIONS = [
    "How much of my tuition is refunded if I withdraw in week four?",
    "What time does the library close on Saturday?",
    "How many items can a postgraduate student borrow?",
    "How long do I have to appeal a refund decision?",
    "What happens if I arrive 40 minutes late to an examination?",
    "How long is a standard accommodation contract?",
]

UNANSWERABLE_QUESTIONS = [
    "How much does a parking permit cost?",
    "Who is the Vice-Chancellor?",
    "What is the pass mark for a dissertation?",
    "Can I bring a guest to stay overnight in halls?",
]


def load_demo_documents() -> dict[str, str]:
    """Read the demo documents as {doc_id: text}."""
    if not DEMO_DIR.exists():
        return {}
    docs: dict[str, str] = {}
    for path in sorted(DEMO_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs[path.stem] = text
    return docs


def ingest_demo(retriever) -> dict[str, int]:
    """Index the demo corpus into a retriever. Returns {doc_id: chunks}."""
    docs = load_demo_documents()
    if not docs:
        return {}
    return retriever.add_documents(docs)
