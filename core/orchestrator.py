"""Orchestrator - wires Modules A through E.

question -> FinalAnswer.

The full path: retrieve (A) -> grade (B, with a bounded corrective loop on bad
retrieval) -> generate with citations (C) -> verify (D) -> repair or abstain
(E). Each variant of the Phase 6 ablation is this pipeline with later stages
switched off, which is why the stages stay independently toggleable.

As of Phase 2, stages A and B are live and C is the plain generator. D and E
raise NotImplementedError and are reached only when their flags are on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from core.evaluator import RetrievalEvaluator
from core.generator import CitationGenerator, PlainGenerator
from core.repair import Repairer
from core.retriever import Retriever
from core.types import Chunk, ChunkLabel, FinalAnswer, RetrievalAction
from core.verifier import NLIVerifier


@dataclass
class PipelineFlags:
    """Which stages are active. Drives the A-E ablation variants."""

    use_evaluator: bool = True       # Module B
    force_citations: bool = True     # Module C
    verify_citations: bool = True    # Module D
    repair_and_abstain: bool = True  # Module E


@dataclass
class RetrievalTrace:
    """What the retrieval + grading stages did, for the ablation record."""

    rounds: int = 0
    corrective_fired: bool = False
    action: str = ""
    labels: list[str] = field(default_factory=list)
    kept_chunk_ids: list[str] = field(default_factory=list)
    abstained_before_generation: bool = False

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "corrective_fired": self.corrective_fired,
            "action": self.action,
            "labels": self.labels,
            "kept_chunk_ids": self.kept_chunk_ids,
            "abstained_before_generation": self.abstained_before_generation,
        }


class Orchestrator:
    """Runs one question through the active stages of the pipeline."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        evaluator: RetrievalEvaluator | None = None,
        generator: PlainGenerator | None = None,   # CitationGenerator is a subclass
        verifier: NLIVerifier | None = None,
        repairer: Repairer | None = None,
        flags: PipelineFlags | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.evaluator = evaluator or RetrievalEvaluator()
        self.flags = flags or PipelineFlags()
        # Variant C onward forces citations; variants A and B use the plain
        # generator, so the ablation compares like with like on everything else.
        self.generator = generator or (
            CitationGenerator() if self.flags.force_citations else PlainGenerator()
        )
        self.verifier = verifier or NLIVerifier()
        self.repairer = repairer or Repairer()

    # -- Modules A + B ---------------------------------------------------
    def retrieve_and_grade(self, question: str) -> tuple[list[Chunk], RetrievalTrace]:
        """Retrieve, grade, and re-retrieve at most `max_corrective_rounds` times.

        The loop is bounded by config. An unbounded re-query on a hard question
        never terminates, and every round costs CPU time even though grading
        itself is free.
        """
        trace = RetrievalTrace()
        chunks = self.retriever.retrieve(question)
        trace.rounds = 1

        if not self.flags.use_evaluator:
            trace.action = "evaluator_disabled"
            trace.kept_chunk_ids = [c.chunk_id for c in chunks]
            return chunks, trace

        cfg = self.evaluator.cfg
        result = self.evaluator.grade(question, chunks)

        rounds_used = 0
        while (
            result.action is RetrievalAction.CORRECTIVE_RETRIEVE
            and rounds_used < cfg.max_corrective_rounds
        ):
            rounds_used += 1
            trace.corrective_fired = True
            trace.rounds += 1
            # Corrective retrieval: widen the net, then re-grade everything
            # seen so far. Deduplicated by chunk id so a repeat does not get
            # counted twice.
            wider = self.retriever.retrieve(question, k=cfg.corrective_fetch_k)
            seen = {c.chunk_id: c for c in chunks}
            for chunk in wider:
                seen.setdefault(chunk.chunk_id, chunk)
            chunks = list(seen.values())
            result = self.evaluator.grade(question, chunks)

        trace.action = result.action.value
        trace.labels = [g.label.value for g in result.graded]

        if result.action is RetrievalAction.ABSTAIN:
            trace.abstained_before_generation = True
            return [], trace

        # Drop chunks graded `wrong`; keep correct and ambiguous, best first.
        kept = sorted(
            (g for g in result.graded if g.label is not ChunkLabel.WRONG),
            key=lambda g: (g.label is not ChunkLabel.CORRECT, -g.confidence),
        )
        chunks = [g.chunk for g in kept][: config.RETRIEVAL.top_k]
        trace.kept_chunk_ids = [c.chunk_id for c in chunks]
        return chunks, trace

    # -- full pipeline ---------------------------------------------------
    def answer(self, question: str) -> FinalAnswer:
        """Run the pipeline end to end for one question."""
        context, trace = self.retrieve_and_grade(question)

        # An abstention here costs nothing: the pipeline stops before the one
        # paid component is ever called.
        if trace.abstained_before_generation or not context:
            return FinalAnswer(
                question=question,
                text=config.REPAIR.abstention_message,
                sentences=[],
                citations=[],
                abstained=True,
                abstain_reason="no chunk graded usable by Module B",
                trace={"retrieval": trace.as_dict()},
            )

        draft = self.generator.generate(question, context)

        if not self.flags.verify_citations:
            return FinalAnswer(
                question=question,
                text=draft.raw_text,
                sentences=draft.sentences,
                citations=context,
                trace={"retrieval": trace.as_dict()},
            )

        report = self.verifier.verify(draft)          # Module D - Phase 4
        if not self.flags.repair_and_abstain:
            return FinalAnswer(
                question=question,
                text=draft.raw_text,
                sentences=draft.sentences,
                citations=context,
                trace={"retrieval": trace.as_dict(), "verified": report.all_supported},
            )

        final = self.repairer.repair(draft, report)   # Module E - Phase 5
        final.trace.setdefault("retrieval", trace.as_dict())
        return final
