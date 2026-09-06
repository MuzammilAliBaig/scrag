"""Module C tests: sentence segmentation, citation parsing, validation.

Written against deliberately malformed output BEFORE trusting any measured
score, per the Phase 3 design note. No API key needed - every test here runs
against the parser, not the model.

The distinction these tests protect: a sentence can fail in two unrelated ways,
and merging them into one number hides both.

  * **parse failure** - the citation is missing or unreadable
  * **invalid id**    - the citation is readable but names a passage that was
                        never retrieved, i.e. the model invented a source
"""

from __future__ import annotations

import pytest

from core.citation import (
    ParsedSentence,
    parse_cited_text,
    parse_sentence,
    split_sentences,
    validate,
)
from core.types import Chunk


def ctx(n: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"Doc::{i}", text=f"passage {i + 1} text", doc_id="Doc")
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Sentence segmentation (3.2)
# --------------------------------------------------------------------------


def test_splits_simple_sentences():
    assert split_sentences("One fact. Two facts. Three.") == [
        "One fact.",
        "Two facts.",
        "Three.",
    ]


def test_does_not_split_on_titles():
    text = "Dr. Ada Lovelace worked with Mr. Babbage. She wrote the first algorithm."
    assert len(split_sentences(text)) == 2


def test_does_not_split_on_initials():
    text = "J. R. R. Tolkien wrote the novel. It was published in 1937."
    assert len(split_sentences(text)) == 2


def test_does_not_split_on_dotted_initialisms():
    text = "She moved to the U.S. in 1962. She became a citizen later."
    assert len(split_sentences(text)) == 2


def test_does_not_split_on_decimals():
    text = "The value is 3.14 exactly. That is pi."
    assert len(split_sentences(text)) == 2


def test_etc_is_treated_as_an_abbreviation_and_under_splits():
    """A documented limitation of the conservative rule, not a bug.

    "etc." can legitimately end a sentence, and the segmenter cannot tell.
    It chooses not to split, which merges two sentences into one coarse
    sentence. That is the safe direction: an under-split sentence keeps the
    citations of both halves, whereas an over-split one strands the second
    half with no citation and fails the Phase 3 gate outright.
    """
    text = "It covers physics, chemistry, etc. The rest is omitted."
    assert len(split_sentences(text)) == 1


def test_keeps_closing_quotes_with_their_sentence():
    parts = split_sentences('He said "it is done." Then he left.')
    assert len(parts) == 2
    assert parts[0].endswith('"')


def test_handles_question_and_exclamation_marks():
    assert len(split_sentences("Who wrote it? Ada did! That is settled.")) == 3


def test_empty_and_whitespace_input():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_single_sentence_without_terminator():
    assert split_sentences("Ada was a mathematician") == ["Ada was a mathematician"]


# --------------------------------------------------------------------------
# Citation parsing (3.3) - the malformed cases
# --------------------------------------------------------------------------


def test_trailing_single_citation():
    p = parse_sentence("Ada was a mathematician [1].")
    assert p.numbers == [1]
    assert p.had_citation
    assert "[1]" not in p.text
    assert p.text == "Ada was a mathematician."


def test_leading_citation_is_still_found():
    """Models put citations at the front about as often as at the end."""
    p = parse_sentence("[2] Ada worked on the Analytical Engine.")
    assert p.numbers == [2]
    assert p.text == "Ada worked on the Analytical Engine."


def test_multiple_bracketed_citations():
    p = parse_sentence("She was a mathematician and writer [1][3].")
    assert p.numbers == [1, 3]


def test_comma_separated_citations_in_one_bracket():
    p = parse_sentence("She was a mathematician and writer [1, 3].")
    assert p.numbers == [1, 3]


def test_mixed_citation_styles_in_one_sentence():
    p = parse_sentence("A claim [1, 2][5] with several markers.")
    assert p.numbers == [1, 2, 5]


def test_duplicate_citations_are_deduplicated_preserving_order():
    p = parse_sentence("A claim [3][1][3].")
    assert p.numbers == [3, 1]


def test_sentence_with_no_citation_is_flagged_not_guessed():
    p = parse_sentence("Ada was a mathematician.")
    assert p.numbers == []
    assert not p.had_citation


def test_missing_closing_bracket_is_not_a_citation():
    """Malformed markers must not be silently accepted as valid."""
    p = parse_sentence("Ada was a mathematician [1.")
    assert p.numbers == []


def test_non_numeric_bracket_content_is_ignored():
    p = parse_sentence("Ada was a mathematician [one] and writer [Doc::0].")
    assert p.numbers == []


def test_punctuation_is_tidied_after_stripping_markers():
    p = parse_sentence("Ada was a mathematician [1] .")
    assert p.text == "Ada was a mathematician."


def test_parse_cited_text_splits_and_parses_each_sentence():
    raw = "Ada was a mathematician [1]. She worked on the Analytical Engine [2]."
    parsed = parse_cited_text(raw)
    assert len(parsed) == 2
    assert parsed[0].numbers == [1]
    assert parsed[1].numbers == [2]


def test_parse_cited_text_flags_the_uncited_sentence_only():
    raw = "Ada was a mathematician [1]. She was also a writer."
    parsed = parse_cited_text(raw)
    assert parsed[0].had_citation
    assert not parsed[1].had_citation


# --------------------------------------------------------------------------
# Validation (3.4) - invented sources are a hard failure
# --------------------------------------------------------------------------


def test_valid_citation_maps_to_the_real_chunk_id():
    parsed = [ParsedSentence("A claim.", "A claim. [2]", [2], True)]
    sentences, stats = validate(parsed, ctx(3))
    assert sentences[0].citation_ids == ["Doc::1"]      # 1-based -> 0-based
    assert sentences[0].parseable
    assert stats.parseable_rate == 1.0
    assert stats.invalid_ids == 0


def test_out_of_range_citation_is_an_invalid_id_not_a_parse_error():
    """Citing passage 9 when 3 were retrieved means an invented source."""
    parsed = [ParsedSentence("A claim.", "A claim. [9]", [9], True)]
    sentences, stats = validate(parsed, ctx(3))
    assert sentences[0].citation_ids == []
    assert not sentences[0].parseable
    assert stats.with_citation == 1        # it DID cite something
    assert stats.with_valid_citation == 0  # but nothing real
    assert stats.invalid_ids == 1
    assert stats.invalid_id_rate == 1.0


def test_zero_is_rejected_because_numbering_is_one_based():
    parsed = [ParsedSentence("A claim.", "A claim. [0]", [0], True)]
    _, stats = validate(parsed, ctx(3))
    assert stats.invalid_ids == 1


def test_a_sentence_that_cannot_be_attributed_is_never_repaired_by_guessing():
    """Do not assign the nearest chunk. Leave it flagged for Module E."""
    parsed = [ParsedSentence("An unsupported claim.", "An unsupported claim.", [], False)]
    sentences, stats = validate(parsed, ctx(3))
    assert sentences[0].citation_ids == []
    assert not sentences[0].parseable
    assert stats.with_citation == 0
    assert stats.invalid_ids == 0          # nothing invented; nothing cited


def test_partially_valid_citations_keep_the_real_ones():
    parsed = [ParsedSentence("A claim.", "A claim. [1][9]", [1, 9], True)]
    sentences, stats = validate(parsed, ctx(3))
    assert sentences[0].citation_ids == ["Doc::0"]
    assert sentences[0].parseable          # one real citation survives
    assert stats.invalid_ids == 1
    assert stats.sentences_with_invalid_id == 1


def test_parseable_rate_counts_only_sentences_with_a_resolvable_citation():
    parsed = [
        ParsedSentence("A.", "A. [1]", [1], True),
        ParsedSentence("B.", "B.", [], False),
        ParsedSentence("C.", "C. [9]", [9], True),
    ]
    _, stats = validate(parsed, ctx(2))
    assert stats.sentences == 3
    assert stats.with_valid_citation == 1
    assert stats.parseable_rate == pytest.approx(1 / 3)
    assert stats.citation_rate == pytest.approx(2 / 3)   # B has none, C cited garbage


def test_empty_draft_does_not_divide_by_zero():
    _, stats = validate([], ctx(3))
    assert stats.parseable_rate == 0.0
    assert stats.invalid_id_rate == 0.0


def test_the_two_failure_modes_are_reported_separately():
    """A missing citation and an invented one must never collapse into one number."""
    parsed = [
        ParsedSentence("No citation.", "No citation.", [], False),
        ParsedSentence("Invented.", "Invented. [42]", [42], True),
    ]
    _, stats = validate(parsed, ctx(2))
    assert stats.with_citation == 1              # only the invented one cited
    assert stats.sentences_with_invalid_id == 1  # and it was invalid
    assert stats.with_valid_citation == 0
    assert stats.parseable_rate == 0.0
