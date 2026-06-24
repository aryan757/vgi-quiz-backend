"""Topic matching — Section 8.

Decides whether a request's `topic` corresponds to something we have good KB coverage for.

Order of operations (cost-aware — most requests must resolve before the embedding call):
  1. topic == "auto"        -> broad/unconstrained MatchResult (optionally domain-inferred).
  2. fast path              -> case-insensitive exact / containment match against canonical
                               topic names. No API call.
  3. embedding fallback     -> embed the incoming topic via OpenAI, compare against the unique
                               topic_embeddings in knowledge_base (deduplicated by distinct
                               topic). Above threshold => confident match to that canonical
                               topic; below => no confident match (routes to LLM-only).

In-process caches:
  - `_topic_name_embedding_cache`: incoming topic strings -> their embedding (avoid re-embedding
    the same query string within a process).
  - `_kb_topic_candidates`: the distinct (canonical_topic, topic_embedding) set from
    knowledge_base, loaded once per process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.db import get_knowledge_base
from app.kb_taxonomy import all_topics, topic_to_domain
from app.services.embeddings import EmbeddingsClient
from app.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    matched: bool                    # True only for a confident match to a canonical topic
    canonical_topic: str | None      # the canonical KB topic name to use from here on
    domain: str | None               # owning domain, if known
    score: float                     # similarity score (1.0 for exact/fast-path, 0.0 for auto)
    method: str                      # "auto" | "fast_path" | "embedding" | "none"
    is_auto: bool = False            # True when topic == "auto" (broad/unconstrained)


class TopicMatcher:
    def __init__(
        self,
        embeddings: EmbeddingsClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings
        self._threshold = self._settings.topic_match_threshold
        self._canonical_topics = all_topics()
        self._topic_domain = topic_to_domain()
        self._topic_name_embedding_cache: dict[str, list[float]] = {}
        self._kb_topic_candidates: list[tuple[str, list[float]]] | None = None

    # --- public API ---------------------------------------------------------

    async def match(
        self, topic: str, description: str | None = None, job_description: str | None = None
    ) -> MatchResult:
        topic_clean = topic.strip()

        # 1. auto -> broad/unconstrained
        if topic_clean.lower() == "auto":
            domain = self._infer_domain(description, job_description)
            return MatchResult(
                matched=False, canonical_topic=None, domain=domain, score=0.0,
                method="auto", is_auto=True,
            )

        # 2. fast path (no API call)
        fast = self._fast_path(topic_clean)
        if fast is not None:
            return MatchResult(
                matched=True, canonical_topic=fast, domain=self._topic_domain.get(fast),
                score=1.0, method="fast_path",
            )

        # 3. embedding fallback
        return await self._embedding_match(topic_clean)

    # --- step 2: fast path --------------------------------------------------

    def fast_path_lookup(self, topic: str) -> str | None:
        """Public, no-API canonical-topic lookup. Used for grounding in the LLM-only branch."""
        return self._fast_path(topic.strip())

    def _fast_path(self, topic: str) -> str | None:
        low = topic.lower()
        # exact (case-insensitive)
        for canon in self._canonical_topics:
            if canon.lower() == low:
                return canon
        # full-string containment (handles "RAG systems" ~ "RAG", "the Transformers guide" ~ "Transformers")
        for canon in self._canonical_topics:
            cl = canon.lower()
            if cl in low or low in cl:
                return canon
        # singular/plural normalisation: strip a trailing 's' from both sides
        # handles "transformer models" ~ "Transformers" (stripping s: "transformer" in "transformer models")
        low_stripped = low.rstrip("s")
        for canon in self._canonical_topics:
            cl_stripped = canon.lower().rstrip("s")
            if cl_stripped and cl_stripped in low or low_stripped in canon.lower():
                return canon
        return None

    # --- step 3: embedding fallback ----------------------------------------

    async def _embedding_match(self, topic: str) -> MatchResult:
        candidates = await self._load_kb_topic_candidates()
        if not candidates:
            logger.info("Embedding match: no KB topic candidates available; no confident match.")
            return MatchResult(False, None, None, 0.0, "none")

        query_emb = await self._embed_topic_name(topic)
        best_topic, best_score = None, -1.0
        for canon, emb in candidates:
            score = cosine_similarity(query_emb, emb)
            if score > best_score:
                best_topic, best_score = canon, score

        if best_topic is not None and best_score >= self._threshold:
            return MatchResult(
                matched=True, canonical_topic=best_topic,
                domain=self._topic_domain.get(best_topic), score=best_score, method="embedding",
            )
        logger.info("Embedding match: best=%s score=%.3f below threshold %.2f",
                    best_topic, best_score, self._threshold)
        return MatchResult(False, None, None, best_score, "none")

    async def _embed_topic_name(self, topic: str) -> list[float]:
        key = topic.lower()
        if key not in self._topic_name_embedding_cache:
            self._topic_name_embedding_cache[key] = await self._client().embed(topic)
        return self._topic_name_embedding_cache[key]

    async def _load_kb_topic_candidates(self) -> list[tuple[str, list[float]]]:
        """Distinct (topic, topic_embedding) pairs from knowledge_base, loaded once."""
        if self._kb_topic_candidates is not None:
            return self._kb_topic_candidates

        candidates: list[tuple[str, list[float]]] = []
        seen: set[str] = set()
        kb = get_knowledge_base()
        # One representative topic_embedding per distinct topic value.
        cursor = kb.find(
            {"topic_embedding": {"$exists": True, "$ne": []}},
            {"topic": 1, "topic_embedding": 1},
        )
        async for doc in cursor:
            t = doc.get("topic")
            emb = doc.get("topic_embedding")
            if t and emb and t not in seen:
                seen.add(t)
                candidates.append((t, emb))
        self._kb_topic_candidates = candidates
        logger.info("Loaded %d distinct KB topic candidates for embedding match.", len(candidates))
        return candidates

    # --- helpers ------------------------------------------------------------

    def _client(self) -> EmbeddingsClient:
        if self._embeddings is None:
            self._embeddings = EmbeddingsClient(self._settings)
        return self._embeddings

    def _infer_domain(self, description: str | None, job_description: str | None) -> str | None:
        """Light keyword-based domain inference for the auto case (best-effort, no API call)."""
        text = " ".join(filter(None, [description, job_description])).lower()
        if not text:
            return None
        hints = {
            "computer_vision": ["vision", "image", "cnn", "yolo", "detection", "segmentation", "opencv"],
            "machine_learning": ["machine learning", "ml ", "regression", "classifier", "clustering", "scikit"],
            "deep_learning": ["deep learning", "neural", "rnn", "lstm", "transformer", "backprop"],
            "genai": ["genai", "llm", "rag", "langchain", "agent", "fine-tun", "embedding", "prompt"],
        }
        best_domain, best_hits = None, 0
        for domain, kws in hints.items():
            hits = sum(1 for kw in kws if kw in text)
            if hits > best_hits:
                best_domain, best_hits = domain, hits
        return best_domain
