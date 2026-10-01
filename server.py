"""Convenient local entry point: ``python server.py``."""

from __future__ import annotations

import uvicorn

from api.config import settings
from api.server import app


if __name__ == "__main__":
    uvicorn.run("server:app", host=settings.api_host, port=settings.api_port, log_level=settings.log_level.lower())
