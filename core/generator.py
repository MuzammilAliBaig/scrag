"""Module C - Citation generator.

q + good ctx -> cited draft, one citation per sentence.
Done when every sentence carries a parseable citation. Built in Phase 3.

This is the ONLY module that calls the Claude API - the single paid component.
Use the official anthropic SDK, prompt caching on the stable prefix, and the
Message Batches API for evaluation runs. See tech-stack.md.
"""

from __future__ import annotations

import config
from core.types import Chunk, CitedDraft, CitedSentence


class CitationGenerator:
    """Generate an answer in which every sentence cites a retrieved chunk."""

    def __init__(self, cfg: config.GeneratorConfig = config.GENERATOR) -> None:
        self.cfg = cfg

    def build_prompt(self, question: str, context: list[Chunk]) -> str:
        """Assemble the citation-forcing prompt. Stable prefix stays cacheable."""
        raise NotImplementedError("Module C - Phase 3")

    def generate(self, question: str, context: list[Chunk]) -> CitedDraft:
        """Produce a cited draft answer for a question from the given context."""
        raise NotImplementedError("Module C - Phase 3")

    def parse_citations(self, raw_text: str, context: list[Chunk]) -> list[CitedSentence]:
        """Split into sentences and attach cited chunk ids.

        Fallback path when structured output is unavailable. The parse-failure
        rate is a reported Phase 3 metric, kept separate from citation
        correctness, which is Module D concern.
        """
        raise NotImplementedError("Module C - Phase 3")
