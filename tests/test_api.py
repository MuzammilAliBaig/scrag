"""Phase 7 API tests: ingest, health, ask, and the error paths.

The error paths matter as much as the happy one here. A quota error during a
live demo must produce a readable message, and an abstention must NOT look like
a failure to any client.

No API key needed - generation is faked or expected to fail.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
import config
from app.api import app, extract_text
from app.schemas import AnswerResponse
from core.retriever import Retriever
from core.types import Chunk, CitedSentence, FinalAnswer


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client backed by a throwaway index, so tests never touch real data."""
    cfg = replace(
        config.RETRIEVAL,
        index_path=tmp_path / "test.faiss",
        metadata_path=tmp_path / "test.meta.json",
    )
    retriever = Retriever(cfg)
    monkeypatch.setattr(api_module, "_retriever", retriever)
    return TestClient(app), retriever


# --------------------------------------------------------------------------
# Upload parsing
# --------------------------------------------------------------------------


def test_text_upload_is_decoded():
    assert extract_text("policy.txt", b"Refunds are issued in 30 days.") == (
        "Refunds are issued in 30 days."
    )


def test_markdown_upload_is_accepted():
    assert "heading" in extract_text("notes.md", b"# heading\n\nbody")


def test_latin1_text_is_decoded_rather_than_rejected():
    assert extract_text("a.txt", "café".encode("latin-1")) != ""


def test_unsupported_file_type_names_what_is_accepted():
    with pytest.raises(ValueError, match="accepted: .pdf, .txt, .md"):
        extract_text("archive.zip", b"PK\x03\x04")


def test_unreadable_pdf_reports_why():
    with pytest.raises(ValueError, match="could not read PDF"):
        extract_text("broken.pdf", b"not a pdf at all")


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_reports_index_readiness(client):
    http, _ = client
    body = http.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_ready"] is False
    assert body["index_chunks"] == 0


def test_health_never_leaks_the_api_key(client):
    http, _ = client
    body = http.get("/health").json()
    assert isinstance(body["api_key_configured"], bool)
    assert not any(isinstance(v, str) and v.startswith("sk-") for v in body.values())


def test_health_reports_model_readiness(client):
    http, _ = client
    body = http.get("/health").json()
    assert body["verifier_checkpoint"] == config.VERIFIER.nli_model
    assert isinstance(body["evaluator_checkpoint_present"], bool)


# --------------------------------------------------------------------------
# /ingest
# --------------------------------------------------------------------------


def test_ingest_indexes_a_text_file_and_reports_chunks(client):
    http, retriever = client
    files = {"files": ("refunds.txt", b"Refunds are issued within 30 working days. " * 20,
                       "text/plain")}
    body = http.post("/ingest", files=files).json()
    assert body["files"][0]["status"] == "indexed"
    assert body["files"][0]["chunks"] > 0
    assert body["index_size"] == retriever.size > 0


def test_ingest_reports_per_file_status_without_failing_the_batch(client):
    """One bad file among good ones must not abort the upload."""
    http, _ = client
    files = [
        ("files", ("good.txt", b"A useful policy document. " * 20, "text/plain")),
        ("files", ("bad.zip", b"PK\x03\x04", "application/zip")),
        ("files", ("empty.txt", b"", "text/plain")),
    ]
    body = http.post("/ingest", files=files).json()
    statuses = {row["filename"]: row["status"] for row in body["files"]}
    assert statuses["good.txt"] == "indexed"
    assert statuses["bad.zip"] == "failed"
    assert statuses["empty.txt"] == "skipped"


def test_reingesting_a_document_replaces_rather_than_duplicates(client):
    http, retriever = client
    payload = b"Original policy text goes here. " * 20
    http.post("/ingest", files={"files": ("doc.txt", payload, "text/plain")})
    first = retriever.size
    http.post("/ingest", files={"files": ("doc.txt", payload, "text/plain")})
    assert retriever.size == first
    assert len(retriever.documents) == 1


def test_demo_corpus_ingests(client):
    http, retriever = client
    response = http.post("/ingest/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks_added"] > 0
    assert body["documents"] >= 4
    assert retriever.size > 0


# --------------------------------------------------------------------------
# /ask
# --------------------------------------------------------------------------


def test_ask_on_an_empty_index_explains_rather_than_crashing(client):
    http, _ = client
    response = http.post("/ask", json={"question": "anything?"})
    assert response.status_code == 409
    assert "index is empty" in response.json()["detail"].lower()


def test_ask_rejects_an_empty_question(client):
    http, _ = client
    assert http.post("/ask", json={"question": ""}).status_code == 422


def test_quota_exhaustion_returns_a_readable_503_not_a_stack_trace(client, monkeypatch):
    """The failure mode most likely to happen mid-demo."""
    http, retriever = client
    http.post("/ingest/demo")

    from core.generator import GenerationError

    def boom(self, question):
        raise GenerationError(
            "API rejected the request (400): Your credit balance is too low to "
            "access the Anthropic API."
        )

    monkeypatch.setattr("core.orchestrator.Orchestrator.answer", boom)
    response = http.post("/ask", json={"question": "What is the refund policy?"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "credit balance is exhausted" in detail
    assert "Traceback" not in detail
    # It must also say what still works, so the demo can continue.
    assert "verification still work" in detail


def test_missing_api_key_returns_a_readable_503(client, monkeypatch):
    http, _ = client
    http.post("/ingest/demo")

    from core.generator import GenerationError

    def boom(self, question):
        raise GenerationError("ANTHROPIC_API_KEY is not set. The Claude API is...")

    monkeypatch.setattr("core.orchestrator.Orchestrator.answer", boom)
    response = http.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_abstention_is_a_200_not_an_error(client, monkeypatch):
    """A refusal is the product working. An error code would teach clients otherwise."""
    http, _ = client
    http.post("/ingest/demo")

    def abstain(self, question):
        return FinalAnswer(
            question=question,
            text="I cannot answer this from the provided sources.",
            sentences=[],
            citations=[],
            abstained=True,
            abstain_reason="no_usable_chunk",
            trace={"abstain_explanation": "No retrieved passage was graded usable."},
        )

    monkeypatch.setattr("core.orchestrator.Orchestrator.answer", abstain)
    response = http.post("/ask", json={"question": "How much is a parking permit?"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["abstain_reason"] == "no_usable_chunk"
    assert body["abstain_explanation"]


def test_answer_carries_per_sentence_citations(client, monkeypatch):
    http, _ = client
    http.post("/ingest/demo")

    chunk = Chunk(chunk_id="refunds-policy::0", text="80 per cent is refunded.",
                  doc_id="refunds-policy")

    def answer(self, question):
        return FinalAnswer(
            question=question,
            text="Eighty per cent of tuition is refunded.",
            sentences=[CitedSentence("Eighty per cent of tuition is refunded.",
                                     ["refunds-policy::0"])],
            citations=[chunk],
            trace={"retrieval": {"action": "proceed", "rounds": 1}},
        )

    monkeypatch.setattr("core.orchestrator.Orchestrator.answer", answer)
    body = http.post("/ask", json={"question": "How much is refunded?"}).json()
    assert body["abstained"] is False
    assert body["sentences"][0]["citation_ids"] == ["refunds-policy::0"]
    assert body["citations"][0]["text"] == "80 per cent is refunded."
    assert body["module_b_action"] == "proceed"


# --------------------------------------------------------------------------
# Schema mapping
# --------------------------------------------------------------------------


def test_repaired_sentences_are_marked_repaired_for_the_ui():
    """A visibly repaired sentence is the most persuasive thing in the demo."""
    answer = FinalAnswer(
        question="q",
        text="A repaired sentence.",
        sentences=[CitedSentence("A repaired sentence.", ["D::0"])],
        citations=[],
        trace={"repair": {"repair_succeeded": 1, "flagged": 1}, "verdicts": []},
    )
    response = AnswerResponse.from_answer(answer)
    assert response.sentences[0].status == "repaired"
    assert response.repair["repair_succeeded"] == 1
