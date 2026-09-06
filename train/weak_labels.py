"""Weak (programmatic) labeling of (question, chunk) pairs for Module B.

CRAG fine-tuned its evaluator on weak relevance signals rather than hand labels:
PopQA supplies a gold subject Wikipedia title per question, which they used as
the positive signal, with negatives randomly sampled from retrieval results that
are lexically similar but not relevant (Yan et al. 2024, §4.2 and Appendix B.3).
SCRAG follows the same approach, so the training signal is comparable.

The rule below is an *approximation* of train/LABELING.md, and the places where
it must diverge are named honestly in `KNOWN_DIVERGENCES`. train/agreement.py
measures how far it actually diverges by comparing it against a human-labeled
slice. Do not describe the output of this module as hand-labeled data.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.retriever import normalize_doc_id
from core.types import Chunk, ChunkLabel, RetrievalAction
from eval.datasets import QAExample
from eval.metrics import normalize_answer

KNOWN_DIVERGENCES = """
1. Rubric case W2 (right subject, irrelevant content -> `wrong`) is labeled
   `ambiguous` here. The rule cannot tell "on-subject but says nothing about the
   queried property" from "on-subject and partially informative" without reading
   for meaning. This inflates `ambiguous` and depresses `wrong`.
2. Rubric case A2 (under-determined subject, e.g. the Easy Living film/song
   collision) is labeled `wrong` here, because the doc title does not match.
   This is the hardest boundary in the rubric and the rule always takes the
   confident side of it.
3. A gold answer string appearing in an off-subject chunk by coincidence -
   common for generic answers like "politician" - is still labeled `wrong`,
   per rubric section 2.2. That is correct by the rubric but means the label
   cannot be recovered from lexical overlap alone.
"""


@dataclass(frozen=True)
class LabeledPair:
    qid: str
    question: str
    chunk_id: str
    chunk_text: str
    label: ChunkLabel
    source: str = "weak"     # "weak" or "human"

    def as_dict(self) -> dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "label": self.label.value,
            "source": self.source,
        }


def _subject_doc_ids(example: QAExample, alias_map: dict[str, str] | None = None) -> set[str]:
    """Doc ids that count as 'the subject page' for this question.

    Wikipedia redirects mean the fetched article can be filed under a canonical
    title different from PopQA's s_wiki_title, so an alias map is consulted when
    one is available.
    """
    titles = {example.subject_title, example.subject}
    if alias_map:
        titles |= {alias_map.get(t, t) for t in titles if t}
    return {normalize_doc_id(t) for t in titles if t}


def weak_label(
    example: QAExample,
    chunk: Chunk,
    alias_map: dict[str, str] | None = None,
) -> ChunkLabel:
    """Label one (question, chunk) pair by rule. See KNOWN_DIVERGENCES."""
    on_subject = chunk.doc_id in _subject_doc_ids(example, alias_map)
    if not on_subject:
        # Wrong subject. Rubric section 2.2, including the name-collision case.
        return ChunkLabel.WRONG

    chunk_norm = normalize_answer(chunk.text)
    states_answer = any(
        normalize_answer(gold) in chunk_norm for gold in example.answers if gold.strip()
    )
    if states_answer:
        return ChunkLabel.CORRECT
    # On-subject but this window does not state the answer. Divergence 1.
    return ChunkLabel.AMBIGUOUS


def derive_action(labels: list[ChunkLabel]) -> RetrievalAction:
    """Query-level action from per-chunk labels. FROZEN - LABELING.md section 5.

    This is the rule that makes our number comparable to CRAG's Table 4, which
    scores the action chosen for a whole retrieved set rather than per-chunk
    grades. Changing it changes the headline number.
    """
    if not labels:
        return RetrievalAction.ABSTAIN
    if any(l is ChunkLabel.CORRECT for l in labels):
        return RetrievalAction.PROCEED              # CRAG: Correct
    if all(l is ChunkLabel.WRONG for l in labels):
        return RetrievalAction.ABSTAIN              # CRAG: Incorrect
    return RetrievalAction.CORRECTIVE_RETRIEVE      # CRAG: Ambiguous


# CRAG's action names, for reporting alongside ours.
ACTION_TO_CRAG = {
    RetrievalAction.PROCEED: "Correct",
    RetrievalAction.CORRECTIVE_RETRIEVE: "Ambiguous",
    RetrievalAction.ABSTAIN: "Incorrect",
}
