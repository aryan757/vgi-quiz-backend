"""Domain matcher — maps a request's `topic` field to one of the 5 known domains.

The request still uses the field name `topic` (API contract unchanged), but the value
is now expected to be a domain name or something close to one.

Matching order (cost-aware):
  1. topic == "auto"    → unconstrained, infer domain from description/job_description if possible
  2. Fast path          → exact match or alias lookup against DOMAINS (no API call)
  3. Embedding fallback → embed the incoming string, compare against domain_embedding values
                          stored in knowledge_base. Above threshold → confident match.
                          Below threshold → no match → full LLM generation (Branch C).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.db import get_knowledge_base
from app.kb_taxonomy import DOMAINS, VALID_DOMAINS, all_domain_aliases
from app.services.embeddings import EmbeddingsClient
from app.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    matched: bool
    canonical_domain: str | None     # one of the 5 VALID_DOMAINS
    score: float
    method: str                      # "auto" | "fast_path" | "embedding" | "none"
    is_auto: bool = False


class TopicMatcher:
    def __init__(
        self,
        embeddings: EmbeddingsClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings
        self._threshold = self._settings.topic_match_threshold
        self._alias_map = all_domain_aliases()
        self._query_emb_cache: dict[str, list[float]] = {}
        self._kb_domain_candidates: list[tuple[str, list[float]]] | None = None

    async def match(
        self, topic: str, description: str | None = None, job_description: str | None = None
    ) -> MatchResult:
        topic_clean = topic.strip()

        # 1. auto
        if topic_clean.lower() == "auto":
            domain = self._infer_domain_from_text(description, job_description)
            return MatchResult(matched=False, canonical_domain=domain, score=0.0,
                               method="auto", is_auto=True)

        # 2. fast path
        fast = self._fast_path(topic_clean)
        if fast:
            return MatchResult(matched=True, canonical_domain=fast, score=1.0, method="fast_path")

        # 3. embedding fallback
        return await self._embedding_match(topic_clean)

    def fast_path_lookup(self, topic: str) -> str | None:
        return self._fast_path(topic.strip())

    # --- fast path ----------------------------------------------------------

    def _fast_path(self, topic: str) -> str | None:
        low = topic.lower().strip()
        # direct alias / exact match
        if low in self._alias_map:
            return self._alias_map[low]
        # partial containment against all aliases
        for alias, domain in self._alias_map.items():
            if alias in low or low in alias:
                return domain
        return None

    # --- embedding fallback -------------------------------------------------

    async def _embedding_match(self, topic: str) -> MatchResult:
        candidates = await self._load_domain_candidates()
        if not candidates:
            return MatchResult(False, None, 0.0, "none")

        query_emb = await self._embed(topic)
        best_domain, best_score = None, -1.0
        for domain, emb in candidates:
            s = cosine_similarity(query_emb, emb)
            if s > best_score:
                best_domain, best_score = domain, s

        if best_domain and best_score >= self._threshold:
            return MatchResult(True, best_domain, best_score, "embedding")

        logger.info("Embedding match: best=%s score=%.3f below threshold %.2f",
                    best_domain, best_score, self._threshold)
        return MatchResult(False, None, best_score, "none")

    async def _embed(self, text: str) -> list[float]:
        key = text.lower()
        if key not in self._query_emb_cache:
            self._query_emb_cache[key] = await self._client().embed(text)
        return self._query_emb_cache[key]

    async def _load_domain_candidates(self) -> list[tuple[str, list[float]]]:
        """One representative domain_embedding per distinct domain in KB."""
        if self._kb_domain_candidates is not None:
            return self._kb_domain_candidates

        candidates: list[tuple[str, list[float]]] = []
        seen: set[str] = set()
        async for doc in get_knowledge_base().find(
            {"domain_embedding": {"$exists": True, "$ne": []}},
            {"domain": 1, "domain_embedding": 1},
        ):
            d = doc.get("domain")
            emb = doc.get("domain_embedding")
            if d and emb and d not in seen:
                seen.add(d)
                candidates.append((d, emb))

        self._kb_domain_candidates = candidates
        logger.info("Loaded %d domain embedding candidates.", len(candidates))
        return candidates

    # --- domain inference ---------------------------------------------------

    def _infer_domain_from_text(self, description: str | None, job_description: str | None) -> str | None:
        text = " ".join(filter(None, [description, job_description])).lower()
        if not text:
            return None
        for domain, meta in DOMAINS.items():
            for alias in meta["aliases"]:
                if alias in text:
                    return domain
        return None

    def _client(self) -> EmbeddingsClient:
        if self._embeddings is None:
            self._embeddings = EmbeddingsClient(self._settings)
        return self._embeddings
