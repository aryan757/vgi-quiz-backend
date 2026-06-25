"""knowledge_base retrieval — filter by domain + difficulty, rank by embedding similarity."""

from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.db import get_knowledge_base
from app.services.embeddings import EmbeddingsClient
from app.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever:
    def __init__(
        self,
        embeddings: EmbeddingsClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings

    async def fetch(
        self,
        domain: str,
        difficulty: str | None,
        limit: int,
        *,
        query_text: str | None = None,
        rank: bool = True,
    ) -> list[dict]:
        """Return up to `limit` KB docs for the given domain/difficulty, ranked by relevance."""
        if limit <= 0:
            return []

        query: dict = {"domain": domain}
        if difficulty:
            query["difficulty"] = difficulty

        pool_size = max(limit * 5, 50)
        candidates = [doc async for doc in get_knowledge_base().find(query).limit(pool_size)]
        if not candidates:
            return []

        if not rank or not query_text:
            return candidates[:limit]

        return (await self._rank(candidates, query_text))[:limit]

    async def grounding_examples(
        self, domain: str, difficulty: str | None, n: int = 3, *, query_text: str | None = None
    ) -> list[dict]:
        return await self.fetch(domain, difficulty, n, query_text=query_text or domain, rank=True)

    async def _rank(self, candidates: list[dict], query_text: str) -> list[dict]:
        usable = [c for c in candidates if c.get("embedding")]
        if not usable:
            return candidates
        try:
            query_emb = await self._client().embed(query_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KB ranking embed failed (%s); returning filter order.", exc)
            return candidates

        usable.sort(key=lambda c: cosine_similarity(query_emb, c["embedding"]), reverse=True)
        no_emb = [c for c in candidates if not c.get("embedding")]
        return usable + no_emb

    def _client(self) -> EmbeddingsClient:
        if self._embeddings is None:
            self._embeddings = EmbeddingsClient(self._settings)
        return self._embeddings
