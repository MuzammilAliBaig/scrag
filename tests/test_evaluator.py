"""Module B unit tests: label rules, thresholds, action derivation, corrective loop.

These run without the fine-tuned checkpoint and without an API key. The parts
that need real model weights are covered by eval/run_evaluator_bench.py, which
is a measurement rather than a test.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import config
from core.evaluator import RetrievalEvaluator
from core.orchestrator import Orchestrator, PipelineFlags
from core.types import Chunk, ChunkLabel, GradedChunk, RetrievalAction
from eval.datasets import QAExample
from train.weak_labels import derive_action, weak_label


def chunk(chunk_id: str, text: str = "text", doc_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id if doc_id is not None else chunk_id.rpartition("::")[0],
    )


def graded(label: ChunkLabel, confidence: float = 0.9, chunk_id: str = "D::0") -> GradedChunk:
    return GradedChunk(chunk=chunk(chunk_id), label=label, confidence=confidence)


# --------------------------------------------------------------------------
# Weak labeling rule (train/weak_labels.py) against the rubric
# --------------------------------------------------------------------------

EXAMPLE = QAExample(
    qid="1",
    question="What is Ada Lovelace's occupation?",
    answers=["mathematician"],
    subject_title="Ada Lovelace",
    subject="Ada Lovelace",
    prop="occupation",
)


def test_on_subject_chunk_stating_the_answer_is_correct():
    c = chunk("Ada_Lovelace::0", "Ada Lovelace was an English mathematician and writer.")
    assert weak_label(EXAMPLE, c) is ChunkLabel.CORRECT


def test_on_subject_chunk_without_the_answer_is_ambiguous():
    c = chunk("Ada_Lovelace::1", "She was the only legitimate child of the poet Lord Byron.")
    assert weak_label(EXAMPLE, c) is ChunkLabel.AMBIGUOUS


def test_off_subject_chunk_is_wrong():
    c = chunk("Mount_Everest::0", "Mount Everest is Earth's highest mountain.")
    assert weak_label(EXAMPLE, c) is ChunkLabel.WRONG


def test_off_subject_chunk_is_wrong_even_when_it_contains_the_gold_string():
    """Rubric section 2.2: wrong subject is wrong, whatever words it contains.

    Generic answers like "politician" or "mathematician" appear on many pages,
    so lexical overlap is not evidence.
    """
    c = chunk("Alan_Turing::0", "Alan Turing was an English mathematician and logician.")
    assert weak_label(EXAMPLE, c) is ChunkLabel.WRONG


def test_answer_matching_ignores_case_and_punctuation():
    c = chunk("Ada_Lovelace::0", "Ada was, above all, a MATHEMATICIAN.")
    assert weak_label(EXAMPLE, c) is ChunkLabel.CORRECT


# --------------------------------------------------------------------------
# Query-level action derivation - FROZEN, LABELING.md section 5
# --------------------------------------------------------------------------


def test_any_correct_chunk_means_proceed():
    labels = [ChunkLabel.WRONG, ChunkLabel.CORRECT, ChunkLabel.WRONG]
    assert derive_action(labels) is RetrievalAction.PROCEED


def test_all_wrong_means_abstain():
    labels = [ChunkLabel.WRONG, ChunkLabel.WRONG]
    assert derive_action(labels) is RetrievalAction.ABSTAIN


def test_ambiguous_without_correct_triggers_corrective_retrieval():
    labels = [ChunkLabel.WRONG, ChunkLabel.AMBIGUOUS]
    assert derive_action(labels) is RetrievalAction.CORRECTIVE_RETRIEVE


def test_empty_retrieval_abstains():
    assert derive_action([]) is RetrievalAction.ABSTAIN


def test_evaluator_action_matches_the_frozen_rule():
    """core.evaluator and train.weak_labels must not drift apart."""
    evaluator = RetrievalEvaluator()
    cases = [
        [ChunkLabel.CORRECT, ChunkLabel.WRONG],
        [ChunkLabel.WRONG, ChunkLabel.WRONG],
        [ChunkLabel.AMBIGUOUS, ChunkLabel.WRONG],
        [ChunkLabel.AMBIGUOUS, ChunkLabel.CORRECT],
    ]
    for labels in cases:
        assert evaluator.decide_action([graded(l) for l in labels]) is derive_action(labels)


# --------------------------------------------------------------------------
# Thresholds (CRAG-style upper / lower band)
# --------------------------------------------------------------------------


def test_high_confidence_correct_is_labeled_correct():
    evaluator = RetrievalEvaluator()
    label, conf = evaluator.label_from_probs(
        {"correct": 0.85, "ambiguous": 0.10, "wrong": 0.05}
    )
    assert label is ChunkLabel.CORRECT
    assert conf == pytest.approx(0.85)


def test_low_correct_probability_falls_to_wrong_when_wrong_dominates():
    evaluator = RetrievalEvaluator()
    label, _ = evaluator.label_from_probs({"correct": 0.05, "ambiguous": 0.15, "wrong": 0.80})
    assert label is ChunkLabel.WRONG


def test_low_correct_probability_falls_to_ambiguous_when_ambiguous_dominates():
    evaluator = RetrievalEvaluator()
    label, _ = evaluator.label_from_probs({"correct": 0.10, "ambiguous": 0.60, "wrong": 0.30})
    assert label is ChunkLabel.AMBIGUOUS


def test_the_middle_band_is_ambiguous():
    """Between the thresholds the model is not confident enough to commit.

    Pinned to explicit thresholds rather than the tuned config values, so
    retuning Module B changes the measured numbers without breaking the test
    that guards the decision logic.
    """
    cfg = replace(config.EVALUATOR, correct_threshold=0.70, wrong_threshold=0.30)
    label, _ = RetrievalEvaluator(cfg).label_from_probs(
        {"correct": 0.50, "ambiguous": 0.25, "wrong": 0.25}
    )
    assert label is ChunkLabel.AMBIGUOUS


def test_thresholds_come_from_config_not_from_hardcoded_numbers():
    strict = replace(config.EVALUATOR, correct_threshold=0.99, wrong_threshold=0.01)
    label, _ = RetrievalEvaluator(strict).label_from_probs(
        {"correct": 0.90, "ambiguous": 0.05, "wrong": 0.05}
    )
    assert label is ChunkLabel.AMBIGUOUS  # 0.90 < 0.99, so no longer `correct`


# --------------------------------------------------------------------------
# Corrective retrieval loop
# --------------------------------------------------------------------------


class FakeRetriever:
    """Returns a fixed set, recording how many times it was queried."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[int | None] = []
        self.cfg = config.RETRIEVAL

    def retrieve(self, question: str, k: int | None = None) -> list[Chunk]:
        self.calls.append(k)
        return list(self._chunks)


class FakeEvaluator:
    """Grades every chunk with a scripted label sequence, one per round."""

    def __init__(self, rounds: list[list[ChunkLabel]]) -> None:
        self.rounds = rounds
        self.cfg = config.EVALUATOR
        self.index = 0

    def grade(self, question, chunks):
        from core.types import EvaluationResult

        labels = self.rounds[min(self.index, len(self.rounds) - 1)]
        self.index += 1
        graded_chunks = [
            GradedChunk(chunk=c, label=l, confidence=0.9)
            for c, l in zip(chunks, labels)
        ]
        return EvaluationResult(graded=graded_chunks, action=derive_action(labels))


def test_corrective_loop_fires_on_ambiguous_and_is_bounded():
    chunks = [chunk("A::0"), chunk("B::0")]
    retriever = FakeRetriever(chunks)
    # Always ambiguous: the loop must stop anyway.
    evaluator = FakeEvaluator([[ChunkLabel.AMBIGUOUS, ChunkLabel.WRONG]])
    orch = Orchestrator(retriever=retriever, evaluator=evaluator)

    _, trace = orch.retrieve_and_grade("q")
    assert trace.corrective_fired
    assert trace.rounds == 1 + config.EVALUATOR.max_corrective_rounds
    assert len(retriever.calls) == trace.rounds


def test_corrective_loop_does_not_fire_when_a_chunk_is_correct():
    retriever = FakeRetriever([chunk("A::0"), chunk("B::0")])
    evaluator = FakeEvaluator([[ChunkLabel.CORRECT, ChunkLabel.WRONG]])
    orch = Orchestrator(retriever=retriever, evaluator=evaluator)

    kept, trace = orch.retrieve_and_grade("q")
    assert not trace.corrective_fired
    assert trace.rounds == 1
    assert [c.chunk_id for c in kept] == ["A::0"]   # the `wrong` chunk is dropped


def test_all_wrong_abstains_before_the_generator_is_called():
    """An abstention must cost nothing - the paid component is never reached."""
    retriever = FakeRetriever([chunk("A::0"), chunk("B::0")])
    evaluator = FakeEvaluator([[ChunkLabel.WRONG, ChunkLabel.WRONG]])

    class ExplodingGenerator:
        def generate(self, *args, **kwargs):
            raise AssertionError("generator must not be called after an abstention")

    orch = Orchestrator(
        retriever=retriever, evaluator=evaluator, generator=ExplodingGenerator()
    )
    answer = orch.answer("q")
    assert answer.abstained
    assert answer.text == config.REPAIR.abstention_message


def test_disabling_the_evaluator_passes_chunks_through_ungraded():
    """Ablation variant A: Module B off, every retrieved chunk reaches the generator."""
    retriever = FakeRetriever([chunk("A::0"), chunk("B::0")])
    orch = Orchestrator(
        retriever=retriever,
        evaluator=FakeEvaluator([[ChunkLabel.WRONG, ChunkLabel.WRONG]]),
        flags=PipelineFlags(use_evaluator=False),
    )
    kept, trace = orch.retrieve_and_grade("q")
    assert len(kept) == 2
    assert trace.action == "evaluator_disabled"
    assert not trace.corrective_fired


def test_kept_chunks_are_ordered_correct_before_ambiguous():
    retriever = FakeRetriever([chunk("A::0"), chunk("B::0"), chunk("C::0")])
    evaluator = FakeEvaluator([[ChunkLabel.AMBIGUOUS, ChunkLabel.CORRECT, ChunkLabel.WRONG]])
    orch = Orchestrator(retriever=retriever, evaluator=evaluator)
    kept, _ = orch.retrieve_and_grade("q")
    assert [c.chunk_id for c in kept] == ["B::0", "A::0"]


# --------------------------------------------------------------------------
# Fail-loud behaviour
# --------------------------------------------------------------------------


def test_missing_checkpoint_raises_rather_than_using_an_untrained_model(tmp_path):
    """An untrained base model would emit random grades that look valid."""
    cfg = replace(config.EVALUATOR, checkpoint_path=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="must not fall back"):
        RetrievalEvaluator(cfg).score_pairs("q", [chunk("A::0")])
