"""Entry point: `python -m app` starts the FastAPI server."""

from __future__ import annotations

import uvicorn

import config


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=config.APP.host,
        port=config.APP.port,
        reload=config.APP.reload,
    )


if __name__ == "__main__":
    main()
