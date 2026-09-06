"""Pydantic request/response schemas for the SCRAG API.

These mirror the dataclasses in core/types.py rather than replacing them. The
pipeline keeps working in plain dataclasses; conversion happens once, at the
edge, in `AnswerResponse.from_answer`.

The response is shaped around what the UI has to render, and two fields carry
most of that weight:

* `abstained` + `abstain_reason` - a refusal is a first-class outcome, not an
  error and not an empty body. The UI renders it as a deliberate product state.
* `sentences[].status` - passed / repaired / dropped. A visibly repaired
  sentence is the most persuasive thing in the demo, so it is surfaced rather
  than hidden.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.types import Chunk, FinalAnswer

SentenceStatus = Literal["passed", "repaired", "dropped", "unverified"]


class ChunkOut(BaseModel):
    """A retrieved passage, as shown when the user clicks a citation."""

    chunk_id: str
    doc_id: str
    text: str
    score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkOut":
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=chunk.score,
        )


class SentenceOut(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    status: SentenceStatus = "unverified"
    entailment_score: float | None = None
    nli_label: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Ablation flags, exposed so the UI can demonstrate what each module adds.
    use_evaluator: bool = True
    force_citations: bool = True
    verify_citations: bool = True
    repair_and_abstain: bool = True


class AnswerResponse(BaseModel):
    question: str
    answer: str
    abstained: bool = False
    abstain_reason: str | None = None
    abstain_explanation: str | None = None
    sentences: list[SentenceOut] = Field(default_factory=list)
    citations: list[ChunkOut] = Field(default_factory=list)
    # What the pipeline did, so the demo can show the machinery working.
    module_b_action: str | None = None
    corrective_fired: bool = False
    retrieval_rounds: int = 0
    repair: dict = Field(default_factory=dict)
    flag_rate: float | None = None
    cost_note: str | None = None

    @classmethod
    def from_answer(cls, answer: FinalAnswer) -> "AnswerResponse":
        trace = answer.trace or {}
        retrieval = trace.get("retrieval") or {}
        repair = trace.get("repair") or {}

        # Map verdicts back onto sentences so each one carries its own status.
        by_text: dict[str, object] = {}
        for verdict in trace.get("verdicts") or []:
            by_text[verdict.sentence.text] = verdict

        repaired_count = repair.get("repair_succeeded", 0)
        sentences: list[SentenceOut] = []
        for sentence in answer.sentences:
            verdict = by_text.get(sentence.text)
            if verdict is None:
                # Present in the final answer but absent from the original
                # verdicts: it is the product of a repair, and Module E only
                # keeps a repair that passed re-verification.
                status: SentenceStatus = "repaired" if repaired_count else "unverified"
                score = None
                label = None
            else:
                status = "passed" if verdict.supported else "unverified"
                score = verdict.entailment_score
                label = getattr(verdict.nli_label, "value", None)
            sentences.append(
                SentenceOut(
                    text=sentence.text,
                    citation_ids=list(sentence.citation_ids),
                    status=status,
                    entailment_score=score,
                    nli_label=label,
                )
            )

        return cls(
            question=answer.question,
            answer=answer.text,
            abstained=answer.abstained,
            abstain_reason=answer.abstain_reason,
            abstain_explanation=trace.get("abstain_explanation"),
            sentences=sentences,
            citations=[ChunkOut.from_chunk(c) for c in answer.citations],
            module_b_action=retrieval.get("action"),
            corrective_fired=bool(retrieval.get("corrective_fired")),
            retrieval_rounds=int(retrieval.get("rounds", 0)),
            repair=repair,
            flag_rate=trace.get("flag_rate"),
            cost_note=trace.get("cost"),
        )


class DroppedSentence(BaseModel):
    """A sentence Module E removed, surfaced so the demo can show the machinery."""

    text: str
    reason: str = "failed verification and could not be repaired"


class IngestFileStatus(BaseModel):
    filename: str
    doc_id: str | None = None
    status: Literal["indexed", "skipped", "failed"]
    chunks: int = 0
    characters: int = 0
    detail: str | None = None


class IngestResponse(BaseModel):
    files: list[IngestFileStatus]
    total_chunks_added: int = 0
    index_size: int = 0
    documents: int = 0
    elapsed_s: float = 0.0
    bulk: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    generator_model: str
    top_k: int
    api_key_configured: bool
    # Index readiness, so the UI can tell "empty index" from "broken service".
    index_ready: bool = False
    index_chunks: int = 0
    index_documents: int = 0
    # Model readiness, reported without loading anything: startup stays fast.
    evaluator_checkpoint_present: bool = False
    verifier_checkpoint: str = ""
    verifier_threshold_calibrated: bool = False
