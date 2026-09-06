"""Sentence segmentation and citation parsing for Module C.

Kept separate from generator.py because this is the part of Phase 3 that has to
be right. The gate is "100% of sentences carry a *parseable* citation", so a
model that formats a citation slightly off makes a perfectly grounded sentence
look ungrounded. Parsing failures and citation-quality failures are different
problems and are counted separately everywhere in this file.

Citations are **passage numbers**, 1-based, matching the order chunks were
presented to the model - the ALCE convention. Chunk ids like
`Ada_Lovelace::3` contain colons and underscores and are miserable for a model
to reproduce exactly; a wrong character there would read as an invented source
rather than a typo. Numbers are mapped back to chunk ids here, where an
out-of-range number is caught as what it is: a hallucinated source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.types import Chunk, CitedSentence

# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------

# Tokens whose trailing period does not end a sentence. Splitting on "Dr." or
# "U.S." would shatter one cited sentence into two, and the second half would
# then look uncited.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon", "gen",
    "col", "lt", "sgt", "capt", "cmdr", "adm", "gov", "sen", "rep", "pres",
    "approx", "est", "fig", "no", "vol", "op", "cf", "eg", "ie", "etc", "al",
    "inc", "ltd", "co", "corp", "dept", "univ", "assn", "bros",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

_SENTENCE_END = re.compile(r"([.!?]+)([\"')\]]*)(\s+)")


def _is_abbreviation(text: str, end: int) -> bool:
    """Whether the period at `end` closes an abbreviation rather than a sentence."""
    prefix = text[:end]
    match = re.search(r"([A-Za-z.]+)\.$", prefix)
    if not match:
        return False
    token = match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # Single letters and dotted initialisms: "J. R. R." or "U.S.A."
    if len(token) == 1:
        return True
    if "." in match.group(1):
        return True
    return False


def _is_decimal(text: str, end: int) -> bool:
    """A period between digits is a decimal point: 3.14, 1.5 million."""
    return end >= 1 and end < len(text) and text[end - 1].isdigit() and text[end].isdigit()


def split_sentences(text: str) -> list[str]:
    """Split into sentences, surviving abbreviations, decimals and quotes.

    Deliberately conservative: when in doubt it does NOT split. An over-split
    sentence loses its citation and counts as a gate failure; an under-split
    one is merely coarse.
    """
    text = " ".join((text or "").split())
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        boundary = match.end(2)          # after punctuation and any closing quote
        period_at = match.start(1)
        if _is_decimal(text, period_at + 1) or _is_abbreviation(text, period_at + 1):
            continue
        piece = text[start:boundary].strip()
        if piece:
            sentences.append(piece)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------
# Citation parsing
# --------------------------------------------------------------------------

# Accepts [1], [1][2], [1, 2], [1,2][3] - anywhere in the sentence, because
# models put them at the front about as often as at the end.
_CITATION_GROUP = re.compile(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]")


@dataclass(frozen=True)
class ParsedSentence:
    """One sentence with the passage numbers it cited, before validation."""

    text: str                    # citation markers stripped
    raw: str                     # exactly as the model wrote it
    numbers: list[int]
    had_citation: bool


def parse_sentence(raw: str) -> ParsedSentence:
    """Extract citation numbers from one sentence and strip the markers."""
    numbers: list[int] = []
    for match in _CITATION_GROUP.finditer(raw):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                numbers.append(int(part))

    stripped = _CITATION_GROUP.sub("", raw)
    stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)   # tidy orphaned punctuation
    stripped = " ".join(stripped.split())

    # Deduplicate, preserving the order the model wrote them in.
    seen: dict[int, None] = {}
    for n in numbers:
        seen.setdefault(n, None)

    return ParsedSentence(
        text=stripped, raw=raw.strip(), numbers=list(seen), had_citation=bool(seen)
    )


def parse_cited_text(raw_text: str) -> list[ParsedSentence]:
    """Fallback path: segment free text and pull citations out of each sentence.

    Used when structured output is unavailable or fails schema validation. The
    rate at which this path is needed is a reported Phase 3 metric.
    """
    return [parse_sentence(s) for s in split_sentences(raw_text)]


# --------------------------------------------------------------------------
# Validation - a cited id that does not exist is a hard failure
# --------------------------------------------------------------------------


@dataclass
class ValidationStats:
    sentences: int = 0
    with_citation: int = 0
    with_valid_citation: int = 0
    invalid_ids: int = 0          # out-of-range passage numbers = invented sources
    sentences_with_invalid_id: int = 0

    @property
    def parseable_rate(self) -> float:
        """The Phase 3 gate: fraction of sentences carrying a usable citation."""
        return self.with_valid_citation / self.sentences if self.sentences else 0.0

    @property
    def citation_rate(self) -> float:
        """Fraction carrying any citation at all, valid or not."""
        return self.with_citation / self.sentences if self.sentences else 0.0

    @property
    def invalid_id_rate(self) -> float:
        return self.sentences_with_invalid_id / self.sentences if self.sentences else 0.0

    def as_dict(self) -> dict:
        return {
            "sentences": self.sentences,
            "with_citation": self.with_citation,
            "with_valid_citation": self.with_valid_citation,
            "invalid_ids": self.invalid_ids,
            "sentences_with_invalid_id": self.sentences_with_invalid_id,
            "parseable_rate": round(self.parseable_rate, 4),
            "citation_rate": round(self.citation_rate, 4),
            "invalid_id_rate": round(self.invalid_id_rate, 4),
        }


def validate(
    parsed: list[ParsedSentence], context: list[Chunk]
) -> tuple[list[CitedSentence], ValidationStats]:
    """Map passage numbers onto real chunk ids, rejecting invented ones.

    A number outside 1..len(context) means the model cited a source that was
    never retrieved. That is a hard failure, not a formatting problem, and the
    sentence is left **unattributed and flagged** - never repaired by guessing
    the nearest chunk. Attaching a plausible-looking source to an unsupported
    sentence is the exact failure this whole project exists to prevent.
    """
    stats = ValidationStats(sentences=len(parsed))
    out: list[CitedSentence] = []

    for sentence in parsed:
        if sentence.had_citation:
            stats.with_citation += 1

        valid_ids: list[str] = []
        invalid_here = 0
        for number in sentence.numbers:
            if 1 <= number <= len(context):
                valid_ids.append(context[number - 1].chunk_id)
            else:
                invalid_here += 1

        stats.invalid_ids += invalid_here
        if invalid_here:
            stats.sentences_with_invalid_id += 1
        if valid_ids:
            stats.with_valid_citation += 1

        out.append(
            CitedSentence(
                text=sentence.text,
                citation_ids=valid_ids,
                # `parseable` means: this sentence carries at least one citation
                # that resolves to a chunk actually retrieved. Whether that
                # chunk *supports* the sentence is Module D's question.
                parseable=bool(valid_ids),
            )
        )

    return out, stats
