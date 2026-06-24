"""knowledge_base retrieval — filter then rank by content-embedding similarity.

Used in two places (Section 9):
  - Step 3 shortfall: pull N questions for (topic, difficulty) to copy into question_bank.
  - Step 4 grounding: pull a few representative examples to steer LLM generation.

Ranking: embed a synthetic query (topic + any available context), score each candidate's
stored content `embedding` by cosine similarity, return the top matches. If no query/context
or no embeddings are present, fall back to plain filter order (no API call).
"""

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
        topic: str,
        difficulty: str | None,
        limit: int,
        *,
        query_text: str | None = None,
        rank: bool = True,
    ) -> list[dict]:
        """Return up to `limit` KB docs for (topic, difficulty), ranked by relevance.

        `difficulty` is the lowercase KB value, or None to match any level.
        """
        if limit <= 0:
            return []

        query: dict = {"topic": topic}
        if difficulty:
            query["difficulty"] = difficulty

        # Over-fetch a pool so ranking has something to choose from, then trim to `limit`.
        pool_size = max(limit * 5, limit)
        candidates = [doc async for doc in get_knowledge_base().find(query).limit(pool_size)]
        if not candidates:
            return []

        if not rank or not query_text:
            return candidates[:limit]

        scored = await self._rank(candidates, query_text)
        return scored[:limit]

    async def grounding_examples(
        self, topic: str, difficulty: str | None, n: int = 3, *, query_text: str | None = None
    ) -> list[dict]:
        """A handful of representative examples for prompt grounding (Step 4)."""
        return await self.fetch(topic, difficulty, n, query_text=query_text or topic, rank=True)

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
        # Append any candidates that had no embedding so we never drop coverage.
        no_emb = [c for c in candidates if not c.get("embedding")]
        return usable + no_emb

    def _client(self) -> EmbeddingsClient:
        if self._embeddings is None:
            self._embeddings = EmbeddingsClient(self._settings)
        return self._embeddings
