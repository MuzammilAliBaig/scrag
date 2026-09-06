"""ALCE loader (ASQA subset) for the Phase 3 citation benchmark.

ALCE ships its own retrieved passages - 100 GTR-retrieved documents per
question - so Module A is deliberately bypassed here. That is the point: the
Phase 3 gate is about the *generator's* citation behaviour, and evaluating it
on ALCE's own retrieval keeps the citation numbers comparable to the ALCE
setting rather than entangled with our corpus.

Data: princeton-nlp/ALCE-data on HuggingFace (ALCE-data.tar, ~450 MB).
Only asqa_eval_gtr_top100.json is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import config
from core.types import Chunk

ALCE_DIR = config.DATA_DIR / "alce" / "ALCE-data"
ASQA_FILE = ALCE_DIR / "asqa_eval_gtr_top100.json"


@dataclass
class ALCEExample:
    sample_id: str
    question: str
    long_answer: str
    # Each sub-question of an ambiguous ASQA question carries its own accepted
    # surface forms; a complete answer should cover all of them.
    short_answer_sets: list[list[str]] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)

    def context(self, k: int) -> list[Chunk]:
        """Top-k ALCE passages as Chunks, with ids traceable to ALCE doc ids."""
        out: list[Chunk] = []
        for i, doc in enumerate(self.docs[:k]):
            title = (doc.get("title") or "").strip()
            text = (doc.get("text") or "").strip()
            out.append(
                Chunk(
                    chunk_id=f"alce-{self.sample_id}::{i}",
                    text=f"{title}\n{text}" if title else text,
                    doc_id=f"alce-{self.sample_id}",
                    metadata={"title": title, "alce_doc_id": str(doc.get("id", ""))},
                )
            )
        return out

    @property
    def all_short_answers(self) -> list[str]:
        return [a for group in self.short_answer_sets for a in group]


def load_asqa(limit: int | None = None, path: Path = ASQA_FILE) -> list[ALCEExample]:
    if not path.exists():
        raise SystemExit(
            f"ALCE data not found at {path}.\n"
            "Download it with:\n"
            "  python -c \"from huggingface_hub import hf_hub_download; "
            "print(hf_hub_download('princeton-nlp/ALCE-data','ALCE-data.tar',repo_type='dataset'))\"\n"
            "then extract the tar into data/alce/."
        )

    rows = json.loads(path.read_text(encoding="utf-8"))
    examples: list[ALCEExample] = []
    for row in rows[: limit or len(rows)]:
        examples.append(
            ALCEExample(
                sample_id=str(row.get("sample_id", len(examples))),
                question=row["question"],
                long_answer=row.get("answer", ""),
                short_answer_sets=[
                    qa.get("short_answers", []) for qa in row.get("qa_pairs", [])
                ],
                docs=row.get("docs", []),
            )
        )
    return examples
