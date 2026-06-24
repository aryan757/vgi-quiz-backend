"""FastAPI application entrypoint for the VGI MCQ Generator."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import close_client, ensure_indexes, ping
from app.routes.generate_questions import router as generate_questions_router

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("vgi_mcq")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Confirm Mongo connectivity and ensure indexes at startup. Don't hard-crash the
    # app if Mongo is briefly unreachable — log loudly; /health reports live status.
    try:
        await ping()
        await ensure_indexes()
        logger.info("Connected to MongoDB and ensured indexes.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Startup MongoDB check failed: %s", exc)
    yield
    await close_client()


app = FastAPI(
    title="VGI MCQ Generator",
    version="2.0.0",
    description="Generates job-readiness MCQs into MongoDB. See MCQ_GENERATOR_BUILD_SPEC_V2.md.",
    lifespan=lifespan,
)

app.include_router(generate_questions_router)


@app.get("/health")
async def health() -> dict:
    """Liveness + Mongo connectivity check."""
    mongo_ok = False
    try:
        mongo_ok = await ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check: Mongo ping failed: %s", exc)
    return {
        "status": "ok" if mongo_ok else "degraded",
        "mongo": "up" if mongo_ok else "down",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }
