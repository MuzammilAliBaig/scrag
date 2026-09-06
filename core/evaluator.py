"""Module B - Retrieval evaluator. THE CONTRIBUTION.

q + chunks -> labels + action.

A fine-tuned DeBERTa-v3-small cross-encoder replacing CRAG's T5-large evaluator.
Runs locally on CPU - no API call on this path, so grading is free no matter how
many chunks are graded.

Two outputs, and the distinction matters for how results are reported:

* **Per-chunk labels** (correct / ambiguous / wrong) drive the pipeline: which
  chunks reach the generator, and whether to re-retrieve.
* **A query-level action** derived from those labels by the frozen rule in
  train/LABELING.md section 5. This is the only output comparable to CRAG's
  reported 84.3% (Yan et al. 2024, Table 4), which scores the action chosen for
  a whole retrieved set, not per-chunk grades.

Thresholding follows CRAG's design: an upper threshold on the relevance score
gives Correct, a lower one gives Incorrect, and the band between them is
Ambiguous - "a retrieval is assumed Correct when the confidence score of at
least one retrieved document is higher than the upper threshold... Incorrect
when the confidence scores of all retrieved documents are below the lower
threshold" (section 4.3).
"""

from __future__ import annotations

import config
from core.types import Chunk, ChunkLabel, EvaluationResult, GradedChunk, RetrievalAction

LABEL_ORDER = [ChunkLabel.CORRECT, ChunkLabel.AMBIGUOUS, ChunkLabel.WRONG]


class RetrievalEvaluator:
    """Grade retrieved chunks correct / ambiguous / wrong, and pick an action."""

    def __init__(self, cfg: config.EvaluatorConfig = config.EVALUATOR) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self.using_finetuned = False

    # -- model -----------------------------------------------------------
    def _load(self) -> None:
        """Load the fine-tuned checkpoint, or fail loudly rather than pretend.

        An untrained base model would emit random 3-class output while looking
        like it works, which is the worst possible failure for the one module
        the project claims as a contribution.
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = self.cfg.checkpoint_path
        if path is None or not (path / "config.json").exists():
            raise FileNotFoundError(
                f"No fine-tuned evaluator at {path}.\n"
                "Module B is the project's contribution and must not fall back to an\n"
                "untrained base model - it would emit random grades that look valid.\n"
                "Train it first:  python -m train.finetune_evaluator"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(
            self._device
        )
        self._model.eval()
        self.using_finetuned = True

    # -- scoring ---------------------------------------------------------
    def score_pairs(self, question: str, chunks: list[Chunk]) -> list[dict[str, float]]:
        """Class probabilities per chunk. Batched, CPU-safe."""
        import torch

        self._load()
        if not chunks:
            return []

        out: list[dict[str, float]] = []
        for start in range(0, len(chunks), self.cfg.batch_size):
            batch = chunks[start : start + self.cfg.batch_size]
            encoded = self._tokenizer(
                [question] * len(batch),
                [c.text for c in batch],
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            id2label = self._model.config.id2label
            for row in probs:
                out.append({id2label[i]: float(p) for i, p in enumerate(row)})
        return out

    # -- grading ---------------------------------------------------------
    def label_from_probs(self, probs: dict[str, float]) -> tuple[ChunkLabel, float]:
        """Apply CRAG-style upper/lower thresholds to p(correct).

        Above the upper threshold the chunk is Correct; at or below the lower
        threshold it is Wrong or Ambiguous by whichever the model prefers; the
        band between is Ambiguous. Both thresholds live in config.py and are
        tuned on the validation split, never on test.
        """
        p_correct = probs.get(ChunkLabel.CORRECT.value, 0.0)
        p_ambiguous = probs.get(ChunkLabel.AMBIGUOUS.value, 0.0)
        p_wrong = probs.get(ChunkLabel.WRONG.value, 0.0)

        if p_correct >= self.cfg.correct_threshold:
            return ChunkLabel.CORRECT, p_correct
        if p_correct <= self.cfg.wrong_threshold:
            if p_wrong >= p_ambiguous:
                return ChunkLabel.WRONG, p_wrong
            return ChunkLabel.AMBIGUOUS, p_ambiguous
        return ChunkLabel.AMBIGUOUS, max(p_ambiguous, p_correct)

    def grade_chunk(self, question: str, chunk: Chunk) -> GradedChunk:
        """Label a single chunk against the question, with a confidence."""
        return self.grade(question, [chunk]).graded[0]

    def grade(self, question: str, chunks: list[Chunk]) -> EvaluationResult:
        """Label every chunk and decide the next retrieval action."""
        scored = self.score_pairs(question, chunks)
        graded = []
        for chunk, probs in zip(chunks, scored):
            label, confidence = self.label_from_probs(probs)
            graded.append(GradedChunk(chunk=chunk, label=label, confidence=confidence))
        return EvaluationResult(graded=graded, action=self.decide_action(graded))

    def decide_action(self, graded: list[GradedChunk]) -> RetrievalAction:
        """Map grades onto proceed / corrective-retrieve / abstain.

        FROZEN - train/LABELING.md section 5. This rule is what makes our number
        comparable to CRAG's Table 4; changing it changes the headline claim.
        """
        if not graded:
            return RetrievalAction.ABSTAIN
        labels = [g.label for g in graded]
        if any(l is ChunkLabel.CORRECT for l in labels):
            return RetrievalAction.PROCEED
        if all(l is ChunkLabel.WRONG for l in labels):
            return RetrievalAction.ABSTAIN
        return RetrievalAction.CORRECTIVE_RETRIEVE
