"""Module D - NLI verifier.

draft -> pass/fail per sentence.
Checks that each cited chunk actually entails the sentence citing it. Done when
unsupported sentences are flagged. Built in Phase 4.

Off-the-shelf NLI checkpoint, run locally. No training, no API cost.
"""

from __future__ import annotations

import config
from core.types import Chunk, CitedDraft, CitedSentence, SentenceVerdict, VerificationReport


class NLIVerifier:
    """Entailment-check every cited sentence against its cited chunks."""

    def __init__(self, cfg: config.VerifierConfig = config.VERIFIER) -> None:
        self.cfg = cfg

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Probability that the premise entails the hypothesis, in [0, 1]."""
        raise NotImplementedError("Module D - Phase 4")

    def verify_sentence(self, sentence: CitedSentence, context: list[Chunk]) -> SentenceVerdict:
        """Pass/fail one sentence against the chunks it cites."""
        raise NotImplementedError("Module D - Phase 4")

    def verify(self, draft: CitedDraft) -> VerificationReport:
        """Verify every sentence in a draft."""
        raise NotImplementedError("Module D - Phase 4")
