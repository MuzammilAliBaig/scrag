"""Module E tests: every abstention trigger fires, and nothing unverified escapes.

The load-bearing test in this file is
`test_no_unverified_sentence_can_reach_the_output`. Everything Modules B, C and
D do is in service of that guarantee, and it is the one property that must hold
on every path, including the odd ones.

No API key and no model weights needed: the generator and verifier are injected
as fakes, so the repair logic is tested in isolation from the models.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import config
from core.repair import AbstainReason, Repairer
from core.types import (
    Chunk,
    CitedDraft,
    CitedSentence,
    NLILabel,
    SentenceVerdict,
    VerificationReport,
)

CONTEXT = [
    Chunk(chunk_id="Doc::0", text="Ada Lovelace was an English mathematician.", doc_id="Doc"),
    Chunk(chunk_id="Doc::1", text="She worked on the Analytical Engine.", doc_id="Doc"),
]


def sentence(text: str, cites: list[str] | None = None, parseable: bool = True) -> CitedSentence:
    return CitedSentence(text=text, citation_ids=cites or ["Doc::0"], parseable=parseable)


def verdict(
    text: str,
    supported: bool,
    cites: list[str] | None = None,
    uncited: bool = False,
    label: NLILabel = NLILabel.ENTAILMENT,
) -> SentenceVerdict:
    return SentenceVerdict(
        sentence=sentence(text, cites, parseable=not uncited),
        supported=supported,
        entailment_score=0.9 if supported else 0.05,
        nli_label=label,
        uncited=uncited,
    )


def draft(sentences: list[CitedSentence], abstained: bool = False) -> CitedDraft:
    return CitedDraft(
        question="Who was Ada Lovelace?",
        sentences=sentences,
        context=CONTEXT,
        raw_text="",
        usage={"abstained": abstained} if abstained else {},
    )


class FakeGenerator:
    """Scripted repair outcomes, so no API call is made."""

    def __init__(self, outcome: str = "repaired", text: str = "A repaired sentence about Ada.") -> None:
        self.outcome = outcome
        self.text = text
        self.calls: list[str] = []

    def regenerate_sentence(self, question, failed_sentence, context):
        self.calls.append(failed_sentence)
        if self.outcome == "unrepairable":
            return None, "unrepairable"
        return CitedSentence(self.text, ["Doc::0"], parseable=True), "repaired"


class FakeVerifier:
    """Verdicts for re-verification of repaired sentences."""

    def __init__(self, passes: bool = True) -> None:
        self.passes = passes
        self.calls = 0

    def verify_sentence(self, sentence, context):
        self.calls += 1
        return SentenceVerdict(
            sentence=sentence,
            supported=self.passes,
            entailment_score=0.9 if self.passes else 0.01,
        )


def repairer(gen_outcome="repaired", recheck_passes=True, cfg=None) -> Repairer:
    return Repairer(
        cfg=cfg or config.REPAIR,
        generator=FakeGenerator(gen_outcome),
        verifier=FakeVerifier(recheck_passes),
    )


# --------------------------------------------------------------------------
# THE guarantee
# --------------------------------------------------------------------------


def test_no_unverified_sentence_can_reach_the_output():
    """The property the whole project exists to provide.

    Exercised across the full cross-product of repair outcomes: every sentence
    in a non-abstained answer must have been either verified originally or
    re-verified after repair.
    """
    cases = [
        [True, True, True],
        [True, False, True],
        [False, True, True],
        [True, True, False],
        [False, False, True],
    ]
    for supported_flags in cases:
        for gen_outcome in ["repaired", "unrepairable"]:
            for recheck in [True, False]:
                verdicts = [
                    verdict(f"Sentence {i}.", ok) for i, ok in enumerate(supported_flags)
                ]
                report = VerificationReport(verdicts=verdicts)
                d = draft([v.sentence for v in verdicts])
                rep = repairer(gen_outcome, recheck)
                answer = rep.repair(d, report)

                if answer.abstained:
                    assert answer.sentences == []
                    assert answer.abstain_reason is not None
                    continue

                originally_supported = {
                    v.sentence.text for v in verdicts if v.supported
                }
                for surviving in answer.sentences:
                    # Either it passed the first time, or it is the repaired
                    # text that passed re-verification.
                    assert (
                        surviving.text in originally_supported
                        or (gen_outcome == "repaired" and recheck)
                    ), f"unverified sentence escaped: {surviving.text!r}"


def test_a_repair_that_fails_reverification_is_dropped_not_kept():
    """A repair never passes by fiat - Module D must agree on the second look."""
    # The survivor must clear the fragment guard, or the abstention path fires
    # first and the drop is never exercised.
    verdicts = [
        verdict("Ada Lovelace was an English mathematician.", True),
        verdict("Bad.", False),
    ]
    rep = repairer(gen_outcome="repaired", recheck_passes=False)
    answer = rep.repair(draft([v.sentence for v in verdicts]),
                        VerificationReport(verdicts=verdicts))
    assert not answer.abstained
    assert [s.text for s in answer.sentences] == [
        "Ada Lovelace was an English mathematician."
    ]
    assert answer.trace["repair"]["dropped"] == 1
    assert answer.trace["repair"]["repair_succeeded"] == 0


# --------------------------------------------------------------------------
# Trigger (a): too much unsupported
# --------------------------------------------------------------------------


def test_trigger_a_fires_when_most_of_the_answer_fails():
    verdicts = [verdict("A.", False), verdict("B.", False), verdict("C.", True)]
    answer = repairer().repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert answer.abstained
    assert answer.abstain_reason == AbstainReason.TOO_MUCH_UNSUPPORTED


def test_trigger_a_does_not_fire_below_the_threshold():
    verdicts = [verdict("A.", True), verdict("B.", True), verdict("C.", False)]
    answer = repairer().repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert not answer.abstained


def test_trigger_a_threshold_comes_from_config():
    verdicts = [verdict("A.", True), verdict("B.", False)]
    strict = replace(config.REPAIR, abstain_if_unsupported_fraction_above=0.1)
    answer = repairer(cfg=strict).repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert answer.abstained
    assert answer.abstain_reason == AbstainReason.TOO_MUCH_UNSUPPORTED


def test_repair_is_not_attempted_when_trigger_a_fires():
    """Abstain before spending API calls on a draft that was mostly wrong."""
    verdicts = [verdict("A.", False), verdict("B.", False)]
    gen = FakeGenerator()
    rep = Repairer(generator=gen, verifier=FakeVerifier())
    rep.repair(draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts))
    assert gen.calls == []


# --------------------------------------------------------------------------
# Trigger (b) and the generator's own refusal
# --------------------------------------------------------------------------


def test_generator_abstention_is_passed_through_not_repaired():
    """A refusal makes no claim, so there is nothing to verify or fix."""
    refusal = sentence(config.REPAIR.abstention_message, [], parseable=False)
    d = draft([refusal], abstained=True)
    report = VerificationReport(verdicts=[verdict(refusal.text, False, [], uncited=True)])
    gen = FakeGenerator()
    answer = Repairer(generator=gen, verifier=FakeVerifier()).repair(d, report)
    assert answer.abstained
    assert answer.abstain_reason == AbstainReason.GENERATOR_ABSTAINED
    assert gen.calls == []          # never tried to repair a refusal


def test_generation_failure_abstains_cleanly():
    d = CitedDraft(
        question="q", sentences=[], context=CONTEXT,
        raw_text="GENERATION FAILED: rate_limited",
    )
    answer = repairer().repair(d, VerificationReport(verdicts=[]))
    assert answer.abstained
    assert answer.abstain_reason in (
        AbstainReason.GENERATION_FAILED, AbstainReason.NOTHING_SURVIVED
    )


# --------------------------------------------------------------------------
# Uncited sentences and the fragment guard
# --------------------------------------------------------------------------


def test_uncited_sentence_is_dropped_without_spending_a_repair_call():
    """Its defect is a Module C failure; regenerating it would waste an API call."""
    verdicts = [
        verdict("Good and long enough to survive.", True),
        verdict("Uncited.", False, cites=[], uncited=True),
    ]
    gen = FakeGenerator()
    answer = Repairer(generator=gen, verifier=FakeVerifier()).repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert gen.calls == []
    assert answer.trace["repair"]["uncited_dropped"] == 1


def test_fragment_after_drops_triggers_abstention():
    """A stub left behind by dropping is worse than a clean refusal."""
    verdicts = [verdict("Yes.", True), verdict("A failing one.", False)]
    answer = repairer(gen_outcome="unrepairable").repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert answer.abstained
    assert answer.abstain_reason == AbstainReason.FRAGMENT_AFTER_DROPS


def test_nothing_surviving_triggers_abstention():
    verdicts = [verdict("A.", False), verdict("B.", True)]
    lenient = replace(config.REPAIR, abstain_if_unsupported_fraction_above=0.99)
    rep = Repairer(
        cfg=lenient, generator=FakeGenerator("unrepairable"), verifier=FakeVerifier()
    )
    # Drop the failing one; the survivor is too short, so this is a fragment.
    answer = rep.repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert answer.abstained


def test_empty_report_abstains():
    answer = repairer().repair(draft([]), VerificationReport(verdicts=[]))
    assert answer.abstained
    assert answer.abstain_reason == AbstainReason.NOTHING_SURVIVED


def test_fragment_thresholds_come_from_config():
    rep = repairer()
    assert rep.is_fragment([sentence("Short.")])
    assert not rep.is_fragment([sentence("A sentence long enough to stand on its own.")])
    lenient = Repairer(cfg=replace(config.REPAIR, min_surviving_chars=1))
    assert not lenient.is_fragment([sentence("Short.")])


# --------------------------------------------------------------------------
# Abstention is a visible state
# --------------------------------------------------------------------------


def test_abstention_carries_a_reason_and_a_message_never_an_empty_response():
    answer = repairer().abstain("q", AbstainReason.NO_USABLE_CHUNK)
    assert answer.abstained
    assert answer.text == config.REPAIR.abstention_message
    assert answer.abstain_reason == AbstainReason.NO_USABLE_CHUNK
    assert answer.trace["abstain_explanation"]
    assert answer.sentences == []
    assert answer.citations == []


def test_every_abstain_reason_has_a_human_readable_explanation():
    codes = [
        v for k, v in vars(AbstainReason).items()
        if isinstance(v, str) and k.isupper() and not k.startswith("_")
    ]
    for code in codes:
        assert code in AbstainReason.HUMAN_READABLE, f"{code} has no explanation"


# --------------------------------------------------------------------------
# Successful answers
# --------------------------------------------------------------------------


def test_a_fully_supported_answer_passes_through_untouched():
    verdicts = [
        verdict("Ada Lovelace was an English mathematician.", True),
        verdict("She worked on the Analytical Engine.", True, cites=["Doc::1"]),
    ]
    gen = FakeGenerator()
    answer = Repairer(generator=gen, verifier=FakeVerifier()).repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert not answer.abstained
    assert len(answer.sentences) == 2
    assert gen.calls == []           # nothing to repair
    assert answer.trace["repair"]["flagged"] == 0


def test_citations_on_the_final_answer_are_only_those_still_cited():
    verdicts = [verdict("Ada Lovelace was a mathematician.", True, cites=["Doc::0"])]
    answer = repairer().repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert [c.chunk_id for c in answer.citations] == ["Doc::0"]


def test_repair_success_is_counted_and_reported():
    verdicts = [
        verdict("A supported sentence that is long enough.", True),
        verdict("A failing sentence.", False),
    ]
    answer = repairer(gen_outcome="repaired", recheck_passes=True).repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    stats = answer.trace["repair"]
    assert stats["repair_attempted"] == 1
    assert stats["repair_succeeded"] == 1
    assert stats["repair_success_rate"] == 1.0
    assert answer.repair_attempts == 1


def test_repair_loop_is_bounded_to_one_attempt_per_sentence():
    verdicts = [verdict("A.", True), verdict("B.", False), verdict("C.", True)]
    gen = FakeGenerator("unrepairable")
    Repairer(generator=gen, verifier=FakeVerifier()).repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert len(gen.calls) == 1       # one failing sentence, one attempt


def test_max_repair_attempts_zero_drops_without_calling_the_generator():
    verdicts = [verdict("A long supported sentence here.", True), verdict("B.", False)]
    no_repair = replace(config.REPAIR, max_repair_attempts=0)
    gen = FakeGenerator()
    answer = Repairer(cfg=no_repair, generator=gen, verifier=FakeVerifier()).repair(
        draft([v.sentence for v in verdicts]), VerificationReport(verdicts=verdicts)
    )
    assert gen.calls == []
    assert answer.trace["repair"]["dropped"] == 1
