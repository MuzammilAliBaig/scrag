"""Module B - Retrieval evaluator. THE CONTRIBUTION.

q + chunks -> labels + action.
A fine-tuned DeBERTa-v3-small replacing the CRAG evaluator. Done when it beats
the CRAG-reported evaluator accuracy and the corrective loop fires. Built in
Phase 2; trained by scripts in train/.

Runs locally. Costs nothing to train or run - no API call on this path.
"""

from __future__ import annotations

import config
from core.types import Chunk, EvaluationResult, GradedChunk, RetrievalAction


class RetrievalEvaluator:
    """Grade retrieved chunks correct / ambiguous / wrong."""

    def __init__(self, cfg: config.EvaluatorConfig = config.EVALUATOR) -> None:
        self.cfg = cfg

    def grade_chunk(self, question: str, chunk: Chunk) -> GradedChunk:
        """Label a single chunk against the question, with a confidence."""
        raise NotImplementedError("Module B - Phase 2")

    def grade(self, question: str, chunks: list[Chunk]) -> EvaluationResult:
        """Label every chunk and decide the next retrieval action."""
        raise NotImplementedError("Module B - Phase 2")

    def decide_action(self, graded: list[GradedChunk]) -> RetrievalAction:
        """Map a set of grades onto proceed / corrective-retrieve / abstain."""
        raise NotImplementedError("Module B - Phase 2")
