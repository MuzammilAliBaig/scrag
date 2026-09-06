"""Module A unit tests: chunker boundaries, index round-trip, retrieval sanity.

No API key needed. The embedding model downloads once on first run (~130 MB)
and is cached by huggingface thereafter.
"""

from __future__ import annotations

import pytest

import config
from core.retriever import Retriever, normalize_doc_id, parse_chunk_id

# --------------------------------------------------------------------------
# Chunker boundaries (1.1)
# --------------------------------------------------------------------------


def test_empty_and_whitespace_documents_produce_no_chunks():
    r = Retriever()
    assert r.chunk("", "doc") == []
    assert r.chunk("   \n\t  ", "doc") == []


def test_short_document_is_a_single_chunk():
    r = Retriever()
    chunks = r.chunk("A short document.", "doc")
    assert len(chunks) == 1
    assert chunks[0].text == "A short document."


def test_chunks_respect_the_configured_size():
    r = Retriever()
    chunks = r.chunk("word " * 2000, "doc")
    assert len(chunks) > 1
    assert all(len(c.text) <= config.RETRIEVAL.chunk_size for c in chunks)


def test_chunks_overlap_so_facts_are_not_split_across_a_boundary():
    r = Retriever()
    chunks = r.chunk("word " * 2000, "doc")
    # Consecutive chunks must start before the previous one ended.
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start_char < previous.end_char


def test_chunker_does_not_split_words():
    r = Retriever()
    text = " ".join(f"token{i}" for i in range(400))
    for chunk in r.chunk(text, "doc"):
        for word in chunk.text.split():
            assert word.startswith("token")
            assert word[5:].isdigit()


def test_overlap_must_be_smaller_than_chunk_size():
    from dataclasses import replace

    bad = replace(config.RETRIEVAL, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        Retriever(bad).chunk("word " * 200, "doc")


# --------------------------------------------------------------------------
# Chunk IDs (1.2) - Module C cites these, Module D resolves them
# --------------------------------------------------------------------------


def test_chunk_ids_are_stable_and_sequential():
    r = Retriever()
    text = "word " * 2000
    first = r.chunk(text, "Some Doc")
    second = r.chunk(text, "Some Doc")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_id for c in first] == [f"Some_Doc::{i}" for i in range(len(first))]


def test_chunk_ids_round_trip_through_the_parser():
    r = Retriever()
    for chunk in r.chunk("word " * 2000, "Ada Lovelace"):
        doc_id, index = parse_chunk_id(chunk.chunk_id)
        assert doc_id == "Ada_Lovelace"
        assert chunk.chunk_id.endswith(f"::{index}")


def test_doc_ids_containing_the_separator_stay_unambiguous():
    assert "::" not in normalize_doc_id("weird::title")
    doc_id, index = parse_chunk_id(Retriever().chunk("text", "weird::title")[0].chunk_id)
    assert doc_id == "weird__title"
    assert index == 0


def test_malformed_chunk_ids_raise():
    for bad in ["no-separator", "doc::notanumber", ""]:
        with pytest.raises(ValueError):
            parse_chunk_id(bad)


# --------------------------------------------------------------------------
# Index round-trip and retrieval (1.3, 1.4, 1.5)
# --------------------------------------------------------------------------

DOCS = {
    "Ada Lovelace": (
        "Ada Lovelace was an English mathematician and writer, known for her work "
        "on the Analytical Engine. She is regarded as the first computer programmer."
    ),
    "Mount Everest": (
        "Mount Everest is Earth's highest mountain above sea level, located in the "
        "Mahalangur Himal sub-range of the Himalayas on the China-Nepal border."
    ),
    "Photosynthesis": (
        "Photosynthesis is the process by which plants convert light energy into "
        "chemical energy, producing oxygen as a by-product."
    ),
}


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """Build a small index in a temp dir, leaving the project index untouched."""
    from dataclasses import replace

    tmp = tmp_path_factory.mktemp("index")
    cfg = replace(
        config.RETRIEVAL,
        index_path=tmp / "test.faiss",
        metadata_path=tmp / "test.meta.json",
    )
    retriever = Retriever(cfg)
    retriever.build_index(DOCS, cache_embeddings=False)
    return retriever, cfg


def test_index_saves_and_loads_without_losing_chunks(built_index):
    retriever, cfg = built_index
    reloaded = Retriever(cfg)
    reloaded.load_index()
    assert reloaded.size == retriever.size
    assert {c.chunk_id for c in reloaded._chunks} == {c.chunk_id for c in retriever._chunks}


def test_loading_a_missing_index_fails_loudly(tmp_path):
    from dataclasses import replace

    cfg = replace(config.RETRIEVAL, index_path=tmp_path / "absent.faiss")
    with pytest.raises(FileNotFoundError):
        Retriever(cfg).load_index()


def test_retrieve_returns_k_results_with_valid_ids(built_index):
    retriever, _ = built_index
    hits = retriever.retrieve("Who was the first computer programmer?", k=2)
    assert len(hits) == 2
    for hit in hits:
        assert hit.score is not None
        parse_chunk_id(hit.chunk_id)               # must parse
        assert retriever.get_chunk(hit.chunk_id)   # must resolve back to text


def test_retrieval_ranks_the_relevant_document_first(built_index):
    retriever, _ = built_index
    top = retriever.retrieve("Who was the first computer programmer?", k=1)[0]
    assert top.doc_id == "Ada_Lovelace"

    top = retriever.retrieve("What is the highest mountain on Earth?", k=1)[0]
    assert top.doc_id == "Mount_Everest"


def test_scores_are_ordered_descending(built_index):
    retriever, _ = built_index
    scores = [h.score for h in retriever.retrieve("plants and light energy", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_the_corpus_is_clamped(built_index):
    retriever, _ = built_index
    assert len(retriever.retrieve("anything", k=999)) == retriever.size


def test_retrieving_before_loading_an_index_fails_loudly():
    with pytest.raises(RuntimeError):
        Retriever().retrieve("query")
