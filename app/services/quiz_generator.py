"""Orchestration — the core 5-step flow for POST /generate-questions (Section 9).

The route is thin; all branching lives here and is unit-testable without FastAPI.

Branches (per the spec):
  A (full reuse)   : confident KB match, question_bank already has >= requested -> reuse, no insert.
  A (shortfall)    : confident KB match, partial inventory -> top up from KB (then LLM if KB short).
  B (KB seed only) : confident KB match, empty inventory -> pull from KB (then LLM if KB short).
  C (LLM only)     : description/job_description present OR no confident match -> full LLM generation.
"""

from __future__ import annotations

import logging

from pymongo.errors import PyMongoError

from app.config import Settings, get_settings
from app.db import get_question_bank
from app.errors import DBError, LLMError
from app.kb_taxonomy import TAXONOMY, subtopics_for
from app.models.enums import Difficulty
from app.models.request import GenerateQuestionsRequest
from app.models.response import GenerateQuestionsResponse
from app.services.kb_retriever import KnowledgeBaseRetriever
from app.services.llm_question_generator import LLMQuestionGenerator
from app.services.topic_matcher import MatchResult, TopicMatcher
from app.utils.transforms import knowledge_base_to_question_bank

logger = logging.getLogger(__name__)


def compute_response_count(existing_reused: int, newly_inserted: int) -> int:
    """PLACEHOLDER SEMANTICS (Section 6.3): total satisfying the request, not strictly
    "newly written." Change here only if "newly inserted only" is confirmed as the intended
    meaning — this is the single audit point for that decision.
    """
    return existing_reused + newly_inserted


class QuizGenerator:
    def __init__(
        self,
        topic_matcher: TopicMatcher | None = None,
        kb_retriever: KnowledgeBaseRetriever | None = None,
        llm_generator: LLMQuestionGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.topic_matcher = topic_matcher or TopicMatcher(settings=self._settings)
        self.kb_retriever = kb_retriever or KnowledgeBaseRetriever(settings=self._settings)
        self.llm_generator = llm_generator or LLMQuestionGenerator(settings=self._settings)

    async def generate_questions(
        self, request: GenerateQuestionsRequest
    ) -> GenerateQuestionsResponse:
        has_context = bool(
            (request.description or "").strip() or (request.job_description or "").strip()
        )

        # Step 1 — branch on description / job_description.
        if has_context:
            logger.info("Branch C (context present) for topic=%s", request.topic)
            reused, inserted = await self._full_llm_generation(request, match=None)
            return self._respond(reused, inserted)

        # Step 2 — topic match check.
        match = await self.topic_matcher.match(
            request.topic, request.description, request.job_description
        )
        if not match.matched:
            logger.info("Branch C (no confident match, method=%s) topic=%s", match.method, request.topic)
            reused, inserted = await self._full_llm_generation(request, match=match)
            return self._respond(reused, inserted)

        # Step 3 — inventory check + fill (Branch A/B).
        logger.info("Branch A/B (confident match -> %s) topic=%s", match.canonical_topic, request.topic)
        reused, inserted = await self._inventory_and_fill(request, match)
        return self._respond(reused, inserted)

    # --- Step 3: confident match path --------------------------------------

    async def _inventory_and_fill(
        self, request: GenerateQuestionsRequest, match: MatchResult
    ) -> tuple[int, int]:
        canonical = match.canonical_topic or request.topic
        qb = get_question_bank()
        diff_filter = self._qb_difficulty_filter(request.difficulty)

        try:
            existing_count = await qb.count_documents({"topic": canonical, **diff_filter})
        except PyMongoError as exc:
            raise DBError(f"Failed to read question_bank inventory: {exc}") from exc

        # Full reuse — nothing to generate.
        if existing_count >= request.question_count:
            return request.question_count, 0  # capped at what was asked for

        shortfall = request.question_count - existing_count
        kb_difficulty = self._kb_difficulty(request.difficulty)
        query_text = f"{canonical} {' '.join(subtopics_for(canonical))}".strip()

        # Pull the shortfall from knowledge_base, ranked.
        kb_docs = await self.kb_retriever.fetch(
            canonical, kb_difficulty, shortfall, query_text=query_text, rank=True
        )
        new_docs = [
            knowledge_base_to_question_bank(d, quiz_type=request.type.value, topic_override=canonical)
            for d in kb_docs
        ]

        # If KB itself can't satisfy the shortfall, top up the remainder via LLM (Step 4 logic).
        remaining = shortfall - len(new_docs)
        if remaining > 0:
            logger.info("KB short by %d for topic=%s; topping up via LLM.", remaining, canonical)
            llm_docs = await self.llm_generator.generate(
                topic=canonical,
                difficulty=request.difficulty,
                count=remaining,
                grounding_examples=kb_docs[:3],
                quiz_type=request.type.value,
            )
            new_docs.extend(llm_docs)

        inserted = await self._insert(new_docs)
        return existing_count, inserted

    # --- Step 4: full LLM generation (Branch C) ----------------------------

    async def _full_llm_generation(
        self, request: GenerateQuestionsRequest, match: MatchResult | None
    ) -> tuple[int, int]:
        # auto topic -> generate broadly across selected topics.
        if request.is_auto_topic:
            domain = match.domain if match else None
            return 0, await self._generate_auto(request, domain)

        # A single concrete topic. Stamp the canonical name if we have a confident/closest match.
        canonical = None
        if match and match.matched:
            canonical = match.canonical_topic
        canonical = canonical or self.topic_matcher.fast_path_lookup(request.topic)
        stamped_topic = canonical or request.topic

        grounding = await self._grounding(canonical, request.difficulty)
        docs = await self.llm_generator.generate(
            topic=stamped_topic,
            difficulty=request.difficulty,
            count=request.question_count,
            description=request.description,
            job_description=request.job_description,
            grounding_examples=grounding,
            quiz_type=request.type.value,
        )
        if not docs:
            raise LLMError("LLM generation returned no valid questions after retry.")
        inserted = await self._insert(docs)
        return 0, inserted

    async def _generate_auto(
        self, request: GenerateQuestionsRequest, domain: str | None
    ) -> int:
        topics = self._select_auto_topics(domain)
        if not topics:
            raise LLMError("No topics available for auto generation.")

        # Split the requested count across the chosen topics.
        n = request.question_count
        per = max(1, n // len(topics))
        plan: list[tuple[str, int]] = []
        allocated = 0
        for i, t in enumerate(topics):
            if allocated >= n:
                break
            take = per if i < len(topics) - 1 else (n - allocated)
            take = min(take, n - allocated)
            plan.append((t, take))
            allocated += take

        all_docs = []
        for topic, take in plan:
            if take <= 0:
                continue
            grounding = await self._grounding(topic, request.difficulty)
            docs = await self.llm_generator.generate(
                topic=topic,
                difficulty=request.difficulty,
                count=take,
                description=request.description,
                job_description=request.job_description,
                grounding_examples=grounding,
                quiz_type=request.type.value,
            )
            all_docs.extend(docs)

        if not all_docs:
            raise LLMError("Auto LLM generation returned no valid questions.")
        return await self._insert(all_docs)

    # --- helpers ------------------------------------------------------------

    async def _grounding(self, canonical: str | None, difficulty: Difficulty) -> list[dict]:
        if not canonical:
            return []
        try:
            return await self.kb_retriever.grounding_examples(
                canonical, self._kb_difficulty(difficulty), n=3
            )
        except Exception as exc:  # noqa: BLE001  (grounding is best-effort)
            logger.warning("Grounding fetch failed for %s: %s", canonical, exc)
            return []

    def _select_auto_topics(self, domain: str | None, limit: int = 5) -> list[str]:
        if domain and domain in TAXONOMY:
            return list(TAXONOMY[domain])[:limit]
        # Spread across domains when no domain signal.
        picks: list[str] = []
        for topics in TAXONOMY.values():
            if topics:
                picks.append(next(iter(topics)))
        return picks[:limit]

    async def _insert(self, docs: list) -> int:
        if not docs:
            return 0
        try:
            payload = [d.to_mongo() for d in docs]
            result = await get_question_bank().insert_many(payload)
            return len(result.inserted_ids)
        except PyMongoError as exc:
            raise DBError(f"Failed to insert into question_bank: {exc}") from exc

    def _qb_difficulty_filter(self, difficulty: Difficulty) -> dict:
        # MIXED matches any concrete level in question_bank.
        if difficulty == Difficulty.MIXED:
            return {}
        return {"difficulty": difficulty.value}

    def _kb_difficulty(self, difficulty: Difficulty) -> str | None:
        # MIXED -> no difficulty constraint when pulling from knowledge_base.
        if difficulty == Difficulty.MIXED:
            return None
        return difficulty.value.lower()

    def _respond(self, reused: int, inserted: int) -> GenerateQuestionsResponse:
        total = compute_response_count(reused, inserted)
        return GenerateQuestionsResponse(
            success=True, message="Questions generated successfully", count=total
        )
