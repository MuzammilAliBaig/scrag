"""Module E - Repair / abstain.

flagged draft -> final answer or abstention.

This is the phase that turns a quality improvement into a guarantee: after
Module E, **no unverified sentence reaches the output**, on any path.

Three outcomes for a flagged sentence, in order:

1. **Repair** - regenerate that one sentence, targeted, then re-verify it
   through Module D. Passing sentences are never touched, because rewriting the
   whole answer would throw away verdicts already earned.
2. **Drop** - if the repair fails verification too, the sentence is removed.
   Bounded at one attempt: an unbounded repair loop burns API budget and can
   oscillate between two equally unsupported phrasings.
3. **Abstain** - if too much of the answer failed, or dropping would leave a
   fragment, the whole answer is replaced by an explicit refusal.

**Abstention is a visible product state, never an empty response.** The
FinalAnswer carries `abstained=True` and a reason naming which trigger fired,
so the API and UI can render it as the deliberate refusal it is.

**An abstention costs less than an answer.** Trigger (b) fires before the
generator is ever called, so the one paid component is skipped entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from core.types import (
    Chunk,
    CitedDraft,
    CitedSentence,
    FinalAnswer,
    SentenceVerdict,
    VerificationReport,
)


class AbstainReason:
    """Why the system refused. Rendered in the UI and kept in the logs."""

    NO_USABLE_CHUNK = "no_usable_chunk"
    TOO_MUCH_UNSUPPORTED = "too_much_unsupported"
    FRAGMENT_AFTER_DROPS = "fragment_after_drops"
    NOTHING_SURVIVED = "nothing_survived"
    GENERATOR_ABSTAINED = "generator_abstained"
    GENERATION_FAILED = "generation_failed"

    HUMAN_READABLE = {
        NO_USABLE_CHUNK: "No retrieved passage was graded usable for this question.",
        TOO_MUCH_UNSUPPORTED: "Too much of the drafted answer could not be verified.",
        FRAGMENT_AFTER_DROPS: "Removing the unverified sentences would leave a fragment.",
        NOTHING_SURVIVED: "No sentence of the drafted answer survived verification.",
        GENERATOR_ABSTAINED: "The passages do not contain the answer.",
        GENERATION_FAILED: "The answer could not be generated.",
    }


@dataclass
class RepairStats:
    """What Module E actually did, for the Phase 6 ablation record."""

    flagged: int = 0
    repair_attempted: int = 0
    repair_succeeded: int = 0
    dropped: int = 0
    uncited_dropped: int = 0

    @property
    def repair_success_rate(self) -> float:
        return self.repair_succeeded / self.repair_attempted if self.repair_attempted else 0.0

    def as_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "dropped": self.dropped,
            "uncited_dropped": self.uncited_dropped,
            "repair_success_rate": round(self.repair_success_rate, 4),
        }


class Repairer:
    """Regenerate, drop, or abstain until nothing unsupported remains."""

    def __init__(
        self,
        cfg: config.RepairConfig = config.REPAIR,
        generator=None,
        verifier=None,
    ) -> None:
        self.cfg = cfg
        self._generator = generator
        self._verifier = verifier

    # Lazily constructed so importing this module needs neither an API key nor
    # the NLI checkpoint on disk.
    @property
    def generator(self):
        if self._generator is None:
            from core.generator import CitationGenerator

            self._generator = CitationGenerator()
        return self._generator

    @property
    def verifier(self):
        if self._verifier is None:
            from core.verifier import NLIVerifier

            self._verifier = NLIVerifier()
        return self._verifier

    # -- abstention triggers (5.4, 5.5) ----------------------------------
    def should_abstain(self, report: VerificationReport) -> tuple[bool, str | None]:
        """Trigger (a): too large a share of the answer failed verification.

        Evaluated on the ORIGINAL verdicts, before repair. A draft that was
        mostly unsupported is not rescued by patching a sentence or two - the
        retrieval or the question was the problem.
        """
        if not report.verdicts:
            return True, AbstainReason.NOTHING_SURVIVED
        if report.flag_rate > self.cfg.abstain_if_unsupported_fraction_above:
            return True, AbstainReason.TOO_MUCH_UNSUPPORTED
        return False, None

    def is_fragment(self, sentences: list[CitedSentence]) -> bool:
        """Minimum surviving-answer condition.

        A single disconnected clause left behind by dropping is worse than a
        clean refusal, so it triggers abstention instead.
        """
        if len(sentences) < self.cfg.min_surviving_sentences:
            return True
        surviving = " ".join(s.text for s in sentences).strip()
        return len(surviving) < self.cfg.min_surviving_chars

    # -- repair loop (5.1, 5.2, 5.3) -------------------------------------
    def repair_sentence(
        self, question: str, verdict: SentenceVerdict, context: list[Chunk]
    ) -> tuple[CitedSentence | None, str]:
        """Regenerate one failed sentence and re-verify it through Module D.

        Returns (sentence, status). A repaired sentence is only accepted if it
        passes verification on the second look - a repair that still fails is a
        drop, never a pass by fiat.
        """
        repaired, status = self.generator.regenerate_sentence(
            question, verdict.sentence.text, context
        )
        if repaired is None:
            return None, status

        recheck = self.verifier.verify_sentence(repaired, context)
        if not recheck.supported:
            return None, "repair_failed_verification"
        return repaired, "repaired"

    # -- entry points ----------------------------------------------------
    def abstain(self, question: str, reason: str, trace: dict | None = None) -> FinalAnswer:
        """Build an abstention carrying the reason that triggered it."""
        return FinalAnswer(
            question=question,
            text=self.cfg.abstention_message,
            sentences=[],
            citations=[],
            abstained=True,
            abstain_reason=reason,
            trace={
                **(trace or {}),
                "abstain_explanation": AbstainReason.HUMAN_READABLE.get(reason, reason),
            },
        )

    def drop_unsupported(
        self, draft: CitedDraft, report: VerificationReport
    ) -> list[CitedSentence]:
        """Remove every sentence that failed verification, repairing nothing."""
        return [v.sentence for v in report.verdicts if v.supported]

    def repair(self, draft: CitedDraft, report: VerificationReport) -> FinalAnswer:
        """Return a fully verified answer, or a clean abstention."""
        question = draft.question
        stats = RepairStats(flagged=len(report.unsupported))

        # The generator already refused. That is a correct outcome, not a
        # defect to repair: a refusal makes no claim, so there is nothing to
        # verify and nothing to fix.
        if draft.usage.get("abstained") or (
            len(draft.sentences) == 1
            and self.cfg.abstention_message.rstrip(".").lower()
            in draft.sentences[0].text.rstrip(".").lower()
        ):
            answer = self.abstain(question, AbstainReason.GENERATOR_ABSTAINED)
            answer.trace["repair"] = stats.as_dict()
            return answer

        if draft.raw_text.startswith("GENERATION FAILED"):
            answer = self.abstain(question, AbstainReason.GENERATION_FAILED)
            answer.trace["repair"] = stats.as_dict()
            return answer

        # Trigger (a), evaluated before spending anything on repair.
        abstain, reason = self.should_abstain(report)
        if abstain:
            answer = self.abstain(question, reason)
            answer.trace["repair"] = stats.as_dict()
            return answer

        surviving: list[CitedSentence] = []
        scores: list[float] = []
        for verdict in report.verdicts:
            if verdict.supported:
                surviving.append(verdict.sentence)
                scores.append(verdict.entailment_score)
                continue

            # An uncited sentence has nothing to repair against, so it is
            # dropped rather than regenerated. Attempting a repair here would
            # spend an API call to rewrite a sentence whose real defect is a
            # Module C failure.
            if verdict.uncited:
                stats.dropped += 1
                stats.uncited_dropped += 1
                continue

            if self.cfg.max_repair_attempts <= 0:
                stats.dropped += 1
                continue

            stats.repair_attempted += 1
            repaired, status = self.repair_sentence(question, verdict, draft.context)
            if repaired is not None:
                stats.repair_succeeded += 1
                surviving.append(repaired)
                scores.append(self.verifier.verify_sentence(repaired, draft.context).entailment_score)
            else:
                stats.dropped += 1

        # Nothing left, or only a fragment: abstain rather than emit a stub.
        if not surviving:
            answer = self.abstain(question, AbstainReason.NOTHING_SURVIVED)
            answer.trace["repair"] = stats.as_dict()
            return answer
        if self.is_fragment(surviving):
            answer = self.abstain(question, AbstainReason.FRAGMENT_AFTER_DROPS)
            answer.trace["repair"] = stats.as_dict()
            return answer

        cited_ids = {cid for s in surviving for cid in s.citation_ids}
        return FinalAnswer(
            question=question,
            text=" ".join(s.text for s in surviving),
            sentences=surviving,
            citations=[c for c in draft.context if c.chunk_id in cited_ids],
            abstained=False,
            repair_attempts=stats.repair_attempted,
            # Per-sentence entailment scores, used to rank answers for the
            # risk-coverage curve in Phase 5 and the ablation in Phase 6.
            trace={"repair": stats.as_dict(), "sentence_scores": scores},
        )
