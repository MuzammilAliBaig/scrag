"""Phase 0 smoke test: the app imports and /health answers 200."""

from __future__ import annotations

from fastapi.testclient import TestClient

import config
from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_reports_config_summary() -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == config.RETRIEVAL.embedding_model
    assert body["generator_model"] == config.GENERATOR.model
    assert body["top_k"] == config.RETRIEVAL.top_k


def test_health_never_leaks_the_api_key() -> None:
    """The one paid dependency's key must never appear in a response."""
    body = client.get("/health").json()
    assert isinstance(body["api_key_configured"], bool)
    assert not any(
        isinstance(v, str) and v.startswith("sk-") for v in body.values()
    )


def test_core_modules_import() -> None:
    """Every Module A-E stub is importable with its real signatures."""
    from core.orchestrator import Orchestrator, PipelineFlags

    orch = Orchestrator()
    assert PipelineFlags().use_evaluator is True
    assert orch.retriever.cfg.top_k == config.RETRIEVAL.top_k
