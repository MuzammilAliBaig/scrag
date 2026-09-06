"""Module E - Repair / abstain.

flagged draft -> final answer or abstention.
Done when no unverified sentence survives. Built in Phase 5.

Abstention is a correct outcome, not a failure, and it costs less than an
answer because the pipeline stops before generating.
"""

from __future__ import annotations

import config
from core.types import CitedDraft, FinalAnswer, VerificationReport


class Repairer:
    """Regenerate, drop, or abstain until nothing unsupported remains."""

    def __init__(self, cfg: config.RepairConfig = config.REPAIR) -> None:
        self.cfg = cfg

    def should_abstain(self, report: VerificationReport) -> bool:
        """Decide whether the draft is beyond repair."""
        raise NotImplementedError("Module E - Phase 5")

    def drop_unsupported(self, draft: CitedDraft, report: VerificationReport) -> CitedDraft:
        """Remove every sentence that failed verification."""
        raise NotImplementedError("Module E - Phase 5")

    def repair(self, draft: CitedDraft, report: VerificationReport) -> FinalAnswer:
        """Return a fully verified answer, or a clean abstention."""
        raise NotImplementedError("Module E - Phase 5")

    def abstain(self, question: str, reason: str) -> FinalAnswer:
        """Build an abstention carrying the reason it was triggered."""
        raise NotImplementedError("Module E - Phase 5")
