"""OpenAI Embeddings wrapper (text-embedding-3-small).

Embeddings are OpenAI-only by deliberate choice, even when LLM_PROVIDER=anthropic.
This is a live dependency: every request that reaches the embedding fallback in topic
matching (Section 8.1) or does KB retrieval ranking calls OpenAI here. Keep the
fast-path string matching in TopicMatcher working so the common case never reaches this.

Both an async client (live request path) and a sync helper (seeding script) are provided.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Thin async wrapper around the OpenAI embeddings endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings but is empty.")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        self.model = self._settings.embedding_model

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


def embed_sync(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Synchronous batch embedding for the seeding script (PyMongo path)."""
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings but is empty.")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    if not texts:
        return []
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in resp.data]
