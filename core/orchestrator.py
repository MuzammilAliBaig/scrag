"""Orchestrator - wires Modules A through E.

question -> FinalAnswer.

The full path: retrieve (A) -> grade (B, with a corrective loop on bad
retrieval) -> generate with citations (C) -> verify (D) -> repair or abstain
(E). Each variant of the Phase 6 ablation is this pipeline with later stages
switched off, which is why the stages stay independently toggleable.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.evaluator import RetrievalEvaluator
from core.generator import CitationGenerator
from core.repair import Repairer
from core.retriever import Retriever
from core.types import FinalAnswer
from core.verifier import NLIVerifier


@dataclass
class PipelineFlags:
    """Which stages are active. Drives the A-E ablation variants."""

    use_evaluator: bool = True       # Module B
    force_citations: bool = True     # Module C
    verify_citations: bool = True    # Module D
    repair_and_abstain: bool = True  # Module E


class Orchestrator:
    """Runs one question through the active stages of the pipeline."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        evaluator: RetrievalEvaluator | None = None,
        generator: CitationGenerator | None = None,
        verifier: NLIVerifier | None = None,
        repairer: Repairer | None = None,
        flags: PipelineFlags | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.evaluator = evaluator or RetrievalEvaluator()
        self.generator = generator or CitationGenerator()
        self.verifier = verifier or NLIVerifier()
        self.repairer = repairer or Repairer()
        self.flags = flags or PipelineFlags()

    def answer(self, question: str) -> FinalAnswer:
        """Run the pipeline end to end for one question."""
        raise NotImplementedError("Orchestrator - wired incrementally, Phases 1-5")
