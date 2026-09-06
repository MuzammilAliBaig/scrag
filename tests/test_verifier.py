"""Module D tests: known entailed / contradicted / neutral triples.

The triples are deliberately unambiguous. If these fail, the checkpoint or the
label mapping is wrong, not the threshold - and a wrong label mapping is the
failure mode that matters most here, because it silently inverts every verdict
while everything still appears to work.

Model-backed tests share one module-scoped verifier so the checkpoint loads
once (~370 MB, CPU).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import config
from core.types import Chunk, CitedDraft, CitedSentence, NLILabel
from core.verifier import NLIVerifier

PREMISE = (
    "Augusta Ada King, Countess of Lovelace, was an English mathematician and "
    "writer, chiefly known for her work on Charles Babbage's proposed mechanical "
    "general-purpose computer, the Analytical Engine. She was the only legitimate "
    "child of the poet Lord Byron."
)

EVEREST = (
    "Mount Everest is Earth's highest mountain above sea level, located in the "
    "Mahalangur Himal sub-range of the Himalayas on the China-Nepal border."
)


@pytest.fixture(scope="module")
def verifier() -> NLIVerifier:
    return NLIVerifier()


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, doc_id=chunk_id.rpartition("::")[0])


# --------------------------------------------------------------------------
# Label mapping - the silent-inversion guard
# --------------------------------------------------------------------------


def test_checkpoint_exposes_all_three_labels(verifier):
    verifier._load()
    assert set(verifier._label_index) == set(NLILabel)


def test_label_indices_are_read_from_the_checkpoint_not_hardcoded(verifier):
    """This checkpoint is (entailment, neutral, contradiction).

    The cross-encoder/nli-* family is (contradiction, entailment, neutral).
    Hardcoding either order breaks silently against the other.
    """
    verifier._load()
    id2label = {int(k): str(v).lower() for k, v in verifier._model.config.id2label.items()}
    for label, index in verifier._label_index.items():
        assert id2label[index] == label.value


def test_probabilities_sum_to_one(verifier):
    scores = verifier.score_pairs([(PREMISE, "Ada Lovelace was a mathematician.")])[0]
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# Known triples
# --------------------------------------------------------------------------


def test_entailed_triple_scores_high(verifier):
    score = verifier.entailment_score(PREMISE, "Ada Lovelace was an English mathematician.")
    assert score > 0.8


def test_contradicted_triple_scores_low_and_labels_contradiction(verifier):
    hypothesis = "Ada Lovelace was Lord Byron's illegitimate child."
    scores = verifier.score_pairs([(PREMISE, hypothesis)])[0]
    assert scores[NLILabel.ENTAILMENT.value] < 0.2
    assert max(scores, key=scores.get) == NLILabel.CONTRADICTION.value


def test_neutral_triple_is_neutral_not_contradiction(verifier):
    """The premise simply does not mention her birth year.

    Neutral means unsupported, NOT false. Collapsing the two would report a
    retrieval failure as a generation failure.
    """
    scores = verifier.score_pairs([(PREMISE, "Ada Lovelace was born in 1815.")])[0]
    assert scores[NLILabel.ENTAILMENT.value] < 0.5
    assert max(scores, key=scores.get) == NLILabel.NEUTRAL.value


def test_unrelated_premise_does_not_entail(verifier):
    score = verifier.entailment_score(EVEREST, "Ada Lovelace was a mathematician.")
    assert score < 0.2


def test_entailed_scores_above_unrelated(verifier):
    """The ordering must hold regardless of where the threshold sits."""
    entailed = verifier.entailment_score(PREMISE, "Ada Lovelace was a writer.")
    unrelated = verifier.entailment_score(EVEREST, "Ada Lovelace was a writer.")
    assert entailed > unrelated


# --------------------------------------------------------------------------
# Verdicts and the gate
# --------------------------------------------------------------------------


def test_every_sentence_receives_a_verdict(verifier):
    """The Phase 4 gate, stated as a test."""
    context = [chunk("Ada::0", PREMISE), chunk("Everest::0", EVEREST)]
    draft = CitedDraft(
        question="Who was Ada Lovelace?",
        sentences=[
            CitedSentence("Ada Lovelace was an English mathematician.", ["Ada::0"]),
            CitedSentence("Ada Lovelace was born in 1815.", ["Ada::0"]),
            CitedSentence("She has no citation.", [], parseable=False),
        ],
        context=context,
    )
    report = verifier.verify(draft)
    assert len(report.verdicts) == len(draft.sentences)
    assert all(isinstance(v.supported, bool) for v in report.verdicts)


def test_supported_sentence_passes_and_unsupported_is_flagged(verifier):
    context = [chunk("Ada::0", PREMISE)]
    draft = CitedDraft(
        question="q",
        sentences=[
            CitedSentence("Ada Lovelace was an English mathematician.", ["Ada::0"]),
            CitedSentence("Ada Lovelace climbed Mount Everest.", ["Ada::0"]),
        ],
        context=context,
    )
    verdicts = verifier.verify(draft).verdicts
    assert verdicts[0].supported
    assert not verdicts[1].supported


def test_uncited_sentence_is_flagged_and_marked_uncited(verifier):
    """No citation means nothing to verify - a Module C failure, not a Module D one."""
    draft = CitedDraft(
        question="q",
        sentences=[CitedSentence("An unattributed claim.", [], parseable=False)],
        context=[chunk("Ada::0", PREMISE)],
    )
    verdict = verifier.verify(draft).verdicts[0]
    assert not verdict.supported
    assert verdict.uncited
    assert verdict.entailment_score == 0.0


def test_citation_naming_a_missing_chunk_is_treated_as_uncited(verifier):
    draft = CitedDraft(
        question="q",
        sentences=[CitedSentence("A claim.", ["NotInContext::7"])],
        context=[chunk("Ada::0", PREMISE)],
    )
    assert verifier.verify(draft).verdicts[0].uncited


def test_per_citation_scores_are_recorded_for_phase_five(verifier):
    """Phase 5 needs per-citation scores to drop the weakest citation."""
    context = [chunk("Ada::0", PREMISE), chunk("Everest::0", EVEREST)]
    draft = CitedDraft(
        question="q",
        sentences=[
            CitedSentence("Ada Lovelace was a mathematician.", ["Ada::0", "Everest::0"])
        ],
        context=context,
    )
    verdict = verifier.verify(draft).verdicts[0]
    assert set(verdict.per_citation) == {"Ada::0", "Everest::0"}
    assert verdict.per_citation["Ada::0"] > verdict.per_citation["Everest::0"]


def test_supporting_chunk_id_points_at_the_best_citation(verifier):
    context = [chunk("Everest::0", EVEREST), chunk("Ada::0", PREMISE)]
    draft = CitedDraft(
        question="q",
        sentences=[
            CitedSentence("Ada Lovelace was a mathematician.", ["Everest::0", "Ada::0"])
        ],
        context=context,
    )
    assert verifier.verify(draft).verdicts[0].supporting_chunk_id == "Ada::0"


# --------------------------------------------------------------------------
# Threshold and policy come from config
# --------------------------------------------------------------------------


def test_threshold_is_read_from_config_not_hardcoded():
    context = [chunk("Ada::0", PREMISE)]
    draft = CitedDraft(
        question="q",
        sentences=[CitedSentence("Ada Lovelace was an English mathematician.", ["Ada::0"])],
        context=context,
    )
    lenient = NLIVerifier(replace(config.VERIFIER, entailment_threshold=0.01))
    strict = NLIVerifier(replace(config.VERIFIER, entailment_threshold=0.999999))
    assert lenient.verify(draft).verdicts[0].supported
    assert not strict.verify(draft).verdicts[0].supported


def test_multi_citation_policy_is_recorded_on_the_verdict(verifier):
    context = [chunk("Ada::0", PREMISE), chunk("Everest::0", EVEREST)]
    draft = CitedDraft(
        question="q",
        sentences=[
            CitedSentence("Ada Lovelace was a mathematician.", ["Ada::0", "Everest::0"])
        ],
        context=context,
    )
    any_policy = NLIVerifier(replace(config.VERIFIER, multi_citation_policy="any"))
    verdict = any_policy.verify(draft).verdicts[0]
    assert verdict.policy == "any"
    # Under `any`, one entailing chunk is enough even with a distractor cited.
    assert verdict.supported


def test_report_separates_contradictions_from_all_unsupported(verifier):
    context = [chunk("Ada::0", PREMISE)]
    draft = CitedDraft(
        question="q",
        sentences=[
            CitedSentence("Ada Lovelace was Lord Byron's illegitimate child.", ["Ada::0"]),
            CitedSentence("Ada Lovelace was born in 1815.", ["Ada::0"]),
        ],
        context=context,
    )
    report = verifier.verify(draft)
    assert len(report.unsupported) == 2
    assert len(report.contradicted) == 1      # only the refuted one
    assert report.flag_rate == 1.0


def test_verdict_log_records_all_three_probabilities(verifier):
    verifier.verdict_log.clear()
    draft = CitedDraft(
        question="q",
        sentences=[CitedSentence("Ada Lovelace was a mathematician.", ["Ada::0"])],
        context=[chunk("Ada::0", PREMISE)],
    )
    verifier.verify(draft)
    row = verifier.verdict_log[-1]
    for key in ["entailment", "neutral", "contradiction", "nli_label", "threshold"]:
        assert key in row
