"""FastAPI application.

Phase 0 ships only /health. Query endpoints arrive in Phase 7 once the
orchestrator is wired.

Startup must stay fast and offline: no model downloads, no index loading here.
"""

from __future__ import annotations

from fastapi import FastAPI

import config

app = FastAPI(title=config.APP.title, version=config.APP.version)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness plus a non-secret config summary. Never returns the API key."""
    return {
        "status": "ok",
        "version": config.APP.version,
        **config.summary(),
        # Whether a key is present, never its value.
        "api_key_configured": config.has_api_key(),
    }
