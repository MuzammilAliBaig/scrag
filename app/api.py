"""FastAPI surface over the SCRAG pipeline.

    uvicorn app.api:app --reload

Three endpoints:

* ``POST /ingest`` - upload documents, chunk, embed, index. Per-file status.
* ``POST /ask``    - run Modules A-E, return the Answer.
* ``GET  /health`` - liveness plus index and model readiness.

The API adapts to the pipeline, never the other way round. In particular an
abstention is returned as a **200 with ``abstained: true``**, not as an error:
a refusal is the product working, and an HTTP error code would teach every
client to treat it as a failure.

Quota exhaustion is handled explicitly. The Claude API is the only paid
component, credit runs out at the worst possible moment, and a stack trace in
front of an audience is avoidable: a 503 with a readable message is not.
"""

from __future__ import annotations

import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import config
from app.schemas import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    IngestFileStatus,
    IngestResponse,
)
from core.generator import GenerationError
from core.orchestrator import Orchestrator, PipelineFlags
from core.retriever import Retriever

logger = logging.getLogger("scrag.api")

app = FastAPI(
    title=config.APP.title,
    version=config.APP.version,
    description="Self-correcting, citation-verified RAG. Answers only from indexed sources.",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}

# One retriever for the process. Loading the FAISS index per request would
# dominate latency, and the index is read-mostly.
_retriever: Retriever | None = None


def app_retrieval_config() -> config.RetrievalConfig:
    """The app indexes into its OWN store, separate from evaluation.

    Uploading a document through the UI must not pollute the corpus every
    measured number in eval/results/ was computed against. Sharing one index
    would silently invalidate the ablation the first time someone tried the
    demo.
    """
    from dataclasses import replace

    return replace(
        config.RETRIEVAL,
        index_path=config.DATA_DIR / "indexes" / "app.faiss",
        metadata_path=config.DATA_DIR / "indexes" / "app.meta.json",
    )


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(app_retrieval_config())
        _retriever.ensure_index()
    return _retriever


# --------------------------------------------------------------------------
# Upload parsing
# --------------------------------------------------------------------------
def extract_text(filename: str, raw: bytes) -> str:
    """Get plain text out of an upload, or raise ValueError with a readable why."""
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:                    # noqa: BLE001
            raise ValueError(f"could not read PDF: {exc}") from exc
        text = "\n\n".join(pages).strip()
        if not text:
            # A scanned PDF has pages but no text layer. Saying so is more use
            # than "0 chunks indexed".
            raise ValueError(
                "PDF contains no extractable text (it may be scanned images; "
                "OCR is out of scope)"
            )
        return text

    if any(lowered.endswith(suffix) for suffix in TEXT_SUFFIXES):
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        raise ValueError("could not decode text file as UTF-8 or Latin-1")

    raise ValueError(
        f"unsupported file type {filename!r}; accepted: .pdf, .txt, .md"
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus index and model readiness. Never returns the API key."""
    retriever = get_retriever()
    checkpoint = config.EVALUATOR.checkpoint_path
    return HealthResponse(
        status="ok",
        version=config.APP.version,
        **config.summary(),
        api_key_configured=config.has_api_key(),
        index_ready=retriever.size > 0,
        index_chunks=retriever.size,
        index_documents=len(retriever.documents),
        evaluator_checkpoint_present=bool(
            checkpoint and (checkpoint / "config.json").exists()
        ),
        verifier_checkpoint=config.VERIFIER.nli_model,
        verifier_threshold_calibrated=config.VERIFIER.threshold_is_calibrated,
    )


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness, distinct from liveness.

    A cold container answers /health immediately but cannot serve a query until
    the embedding model is resident. Free-tier cold starts take tens of
    seconds, and a warming app that looks identical to a hung one is the
    difference between "slow" and "broken" to whoever is watching.

    Returns 200 when a query can be served, 503 while warming.
    """
    retriever = get_retriever()
    index_ready = retriever.size > 0
    models_ready = retriever._model is not None

    ready_now = index_ready and models_ready
    return JSONResponse(
        status_code=200 if ready_now else 503,
        content={
            "ready": ready_now,
            "index_ready": index_ready,
            "index_chunks": retriever.size,
            "embedding_model_loaded": models_ready,
            "detail": (
                "ready"
                if ready_now
                else (
                    "index is empty - load the demo corpus or upload documents"
                    if not index_ready
                    else "warming up: the embedding model is still loading"
                )
            ),
        },
    )


@app.on_event("startup")
def warm_up() -> None:
    """Load the embedding model at startup rather than on the first query.

    Without this the first visitor pays the model-load cost on top of the cold
    start and assumes the app is broken. Failure here is logged, not fatal:
    the app should still come up and report itself unready.
    """
    try:
        retriever = get_retriever()
        retriever.embed(["warm up"])
        logger.info("warm-up complete: %d chunks indexed", retriever.size)
    except Exception:                               # noqa: BLE001
        logger.exception("warm-up failed; /ready will report not-ready")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    """Upload documents, chunk, embed and index them. Per-file status.

    One bad file does not fail the batch: each file gets its own status so a
    scanned PDF among five good documents is reported rather than aborting the
    upload.
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    retriever = get_retriever()
    statuses: list[IngestFileStatus] = []
    to_index: dict[str, str] = {}

    for upload in files:
        raw = await upload.read()
        name = upload.filename or "unnamed"

        if not raw:
            statuses.append(
                IngestFileStatus(filename=name, status="skipped", detail="file is empty")
            )
            continue
        if len(raw) > MAX_UPLOAD_BYTES:
            statuses.append(
                IngestFileStatus(
                    filename=name,
                    status="skipped",
                    detail=f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            )
            continue

        try:
            text = extract_text(name, raw)
        except ValueError as exc:
            statuses.append(
                IngestFileStatus(filename=name, status="failed", detail=str(exc))
            )
            continue

        doc_id = name.rsplit(".", 1)[0]
        to_index[doc_id] = text
        statuses.append(
            IngestFileStatus(
                filename=name, doc_id=doc_id, status="indexed", characters=len(text)
            )
        )

    if to_index:
        try:
            added = retriever.add_documents(to_index)
        except Exception as exc:                    # noqa: BLE001
            logger.exception("indexing failed")
            raise HTTPException(status_code=500, detail=f"indexing failed: {exc}") from exc
        for status in statuses:
            if status.doc_id in added:
                status.chunks = added[status.doc_id]

    return IngestResponse(
        files=statuses,
        total_chunks_added=sum(s.chunks for s in statuses),
        index_size=retriever.size,
        documents=len(retriever.documents),
    )


@app.post("/ingest/demo", response_model=IngestResponse)
def ingest_demo_corpus() -> IngestResponse:
    """Index the bundled demo corpus, so the app is never empty on first open."""
    from app.demo_corpus import ingest_demo, load_demo_documents

    docs = load_demo_documents()
    if not docs:
        raise HTTPException(status_code=404, detail="no demo corpus found in data/demo/")

    retriever = get_retriever()
    added = ingest_demo(retriever)
    return IngestResponse(
        files=[
            IngestFileStatus(
                filename=f"{doc_id}.md",
                doc_id=doc_id,
                status="indexed",
                chunks=added.get(doc_id, 0),
                characters=len(text),
            )
            for doc_id, text in docs.items()
        ],
        total_chunks_added=sum(added.values()),
        index_size=retriever.size,
        documents=len(retriever.documents),
    )


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    """Run the full A-E pipeline on one question.

    An abstention comes back as 200 with ``abstained: true``. It is the system
    working as designed, and returning an error code for it would train every
    client to treat a correct refusal as a fault.
    """
    retriever = get_retriever()
    if retriever.size == 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "The index is empty. Upload documents via /ingest, or load the demo "
                "corpus, before asking a question."
            ),
        )

    orch = Orchestrator(
        retriever=retriever,
        flags=PipelineFlags(
            use_evaluator=request.use_evaluator,
            force_citations=request.force_citations,
            verify_citations=request.verify_citations,
            repair_and_abstain=request.repair_and_abstain,
        ),
    )

    try:
        answer = orch.answer(request.question)
    except GenerationError as exc:
        message = str(exc)
        # The Claude API is the only paid component, and credit runs out at the
        # worst possible moment. A readable 503 beats a stack trace on a
        # projector.
        if "credit balance is too low" in message:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The Claude API credit balance is exhausted, so no new answer can "
                    "be generated. Retrieval, grading and verification still work; "
                    "only generation is blocked. Add credit and retry."
                ),
            ) from exc
        if "ANTHROPIC_API_KEY is not set" in message:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ANTHROPIC_API_KEY is not configured on the server. Generation is "
                    "unavailable until it is set."
                ),
            ) from exc
        raise HTTPException(status_code=502, detail=f"generation failed: {message}") from exc
    except FileNotFoundError as exc:
        # A missing Module B checkpoint, most likely.
        raise HTTPException(
            status_code=503,
            detail=f"a required model is missing on the server: {exc}",
        ) from exc

    return AnswerResponse.from_answer(answer)


@app.exception_handler(Exception)
async def unhandled(request, exc):            # noqa: ANN001, ARG001
    """Last resort: log the trace, show the user a sentence."""
    logger.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {type(exc).__name__}"},
    )
