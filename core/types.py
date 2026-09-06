"""Shared data types crossing the Module A-E boundaries.

Defined once here so each phase fills in a body rather than redesigning an
interface. Every type names the module that produces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of source text. Produced by Module A."""

    chunk_id: str
    text: str
    doc_id: str
    # Position of this chunk within its document, for provenance display.
    start_char: int = 0
    end_char: int = 0
    score: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class ChunkLabel(str, Enum):
    """Module B's per-chunk grade."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    WRONG = "wrong"


class RetrievalAction(str, Enum):
    """What Module B tells the orchestrator to do next."""

    PROCEED = "proceed"
    CORRECTIVE_RETRIEVE = "corrective_retrieve"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class GradedChunk:
    chunk: Chunk
    label: ChunkLabel
    confidence: float


@dataclass(frozen=True)
class EvaluationResult:
    """Module B output: labels plus the action they imply."""

    graded: list[GradedChunk]
    action: RetrievalAction

    @property
    def usable_chunks(self) -> list[Chunk]:
        return [g.chunk for g in self.graded if g.label is not ChunkLabel.WRONG]


@dataclass(frozen=True)
class CitedSentence:
    """One sentence of a draft answer with the chunk ids it claims support it."""

    text: str
    citation_ids: list[str]
    # False when the citation could not be parsed at all — a Phase 3 metric
    # distinct from the citation being wrong, which is Phase 4's problem.
    parseable: bool = True


@dataclass(frozen=True)
class CitedDraft:
    """Module C output."""

    question: str
    sentences: list[CitedSentence]
    context: list[Chunk]
    raw_text: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def parseable_rate(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(s.parseable for s in self.sentences) / len(self.sentences)


@dataclass(frozen=True)
class SentenceVerdict:
    """Module D output, one per sentence."""

    sentence: CitedSentence
    supported: bool
    entailment_score: float
    supporting_chunk_id: str | None = None


@dataclass(frozen=True)
class VerificationReport:
    verdicts: list[SentenceVerdict]

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [v for v in self.verdicts if not v.supported]

    @property
    def all_supported(self) -> bool:
        return all(v.supported for v in self.verdicts)


@dataclass(frozen=True)
class FinalAnswer:
    """Module E output. `abstained` is a first-class outcome, not a failure."""

    question: str
    text: str
    sentences: list[CitedSentence]
    citations: list[Chunk]
    abstained: bool = False
    abstain_reason: str | None = None
    repair_attempts: int = 0
    trace: dict[str, object] = field(default_factory=dict)
