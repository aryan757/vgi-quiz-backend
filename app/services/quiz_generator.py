"""Orchestration — the core 5-step flow for POST /generate-questions (Section 9).

The `topic` field in the request now maps to a domain name
(computer_vision / machine_learning / deep_learning / genai / ai_fundamentals).
The frontend-facing question_bank stores the domain name in its `topic` field.

Branches:
  A (full reuse)   : confident domain match, question_bank already has >= requested
  A (shortfall)    : confident domain match, partial inventory → top up from KB
  B (KB seed)      : confident domain match, empty inventory → pull from KB
  C (LLM only)     : description/job_description present OR no confident domain match
"""

from __future__ import annotations

import logging

from pymongo.errors import PyMongoError

from app.config import Settings, get_settings
from app.db import get_question_bank
from app.errors import DBError, LLMError
from app.kb_taxonomy import VALID_DOMAINS, domain_description
from app.models.enums import Difficulty
from app.models.request import GenerateQuestionsRequest
from app.models.response import GenerateQuestionsResponse
from app.services.kb_retriever import KnowledgeBaseRetriever
from app.services.llm_question_generator import LLMQuestionGenerator
from app.services.topic_matcher import MatchResult, TopicMatcher
from app.utils.transforms import knowledge_base_to_question_bank

logger = logging.getLogger(__name__)


def compute_response_count(existing_reused: int, newly_inserted: int) -> int:
    # PLACEHOLDER SEMANTICS: total satisfying the request, not strictly "newly written."
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

        # Step 1 — description/job_description present → straight to LLM
        if has_context:
            logger.info("Branch C (context present) for domain=%s", request.topic)
            reused, inserted = await self._full_llm_generation(request, match=None)
            return self._respond(reused, inserted)

        # Step 2 — domain match check
        match = await self.topic_matcher.match(
            request.topic, request.description, request.job_description
        )
        if not match.matched:
            logger.info("Branch C (no confident match, method=%s) topic=%s",
                        match.method, request.topic)
            reused, inserted = await self._full_llm_generation(request, match=match)
            return self._respond(reused, inserted)

        # Step 3 — inventory check + fill (Branch A/B)
        logger.info("Branch A/B (domain=%s)", match.canonical_domain)
        reused, inserted = await self._inventory_and_fill(request, match)
        return self._respond(reused, inserted)

    # --- Step 3 -------------------------------------------------------------

    async def _inventory_and_fill(
        self, request: GenerateQuestionsRequest, match: MatchResult
    ) -> tuple[int, int]:
        domain = match.canonical_domain or request.topic
        qb = get_question_bank()
        diff_filter = self._qb_difficulty_filter(request.difficulty)

        try:
            existing_count = await qb.count_documents({"topic": domain, **diff_filter})
        except PyMongoError as exc:
            raise DBError(f"Failed to read question_bank inventory: {exc}") from exc

        if existing_count >= request.question_count:
            return request.question_count, 0

        shortfall = request.question_count - existing_count
        kb_difficulty = self._kb_difficulty(request.difficulty)

        kb_docs = await self.kb_retriever.fetch(
            domain, kb_difficulty, shortfall,
            query_text=domain_description(domain), rank=True
        )
        new_docs = [knowledge_base_to_question_bank(d, quiz_type=request.type.value) for d in kb_docs]

        remaining = shortfall - len(new_docs)
        if remaining > 0:
            logger.info("KB short by %d for domain=%s; topping up via LLM.", remaining, domain)
            llm_docs = await self.llm_generator.generate(
                topic=domain,
                difficulty=request.difficulty,
                count=remaining,
                grounding_examples=kb_docs[:3],
                quiz_type=request.type.value,
            )
            new_docs.extend(llm_docs)

        inserted = await self._insert(new_docs)
        return existing_count, inserted

    # --- Step 4 -------------------------------------------------------------

    async def _full_llm_generation(
        self, request: GenerateQuestionsRequest, match: MatchResult | None
    ) -> tuple[int, int]:
        if request.is_auto_topic:
            domain = (match.canonical_domain if match else None) or \
                     self.topic_matcher._infer_domain_from_text(
                         request.description, request.job_description
                     )
            domains = [domain] if domain else VALID_DOMAINS[:3]
            return 0, await self._generate_across_domains(request, domains)

        # Resolve domain: match result → fast path → use raw topic as label
        domain = None
        if match and match.matched:
            domain = match.canonical_domain
        domain = domain or self.topic_matcher.fast_path_lookup(request.topic) or request.topic

        grounding = await self._grounding(domain, request.difficulty)
        docs = await self.llm_generator.generate(
            topic=domain,
            difficulty=request.difficulty,
            count=request.question_count,
            description=request.description,
            job_description=request.job_description,
            grounding_examples=grounding,
            quiz_type=request.type.value,
        )
        if not docs:
            raise LLMError("LLM generation returned no valid questions after retry.")
        return 0, await self._insert(docs)

    async def _generate_across_domains(
        self, request: GenerateQuestionsRequest, domains: list[str]
    ) -> int:
        n = request.question_count
        per = max(1, n // len(domains))
        all_docs = []
        allocated = 0
        for i, domain in enumerate(domains):
            if allocated >= n:
                break
            take = per if i < len(domains) - 1 else (n - allocated)
            take = min(take, n - allocated)
            grounding = await self._grounding(domain, request.difficulty)
            docs = await self.llm_generator.generate(
                topic=domain, difficulty=request.difficulty, count=take,
                description=request.description, job_description=request.job_description,
                grounding_examples=grounding, quiz_type=request.type.value,
            )
            all_docs.extend(docs)
            allocated += take
        if not all_docs:
            raise LLMError("Auto LLM generation returned no valid questions.")
        return await self._insert(all_docs)

    # --- helpers ------------------------------------------------------------

    async def _grounding(self, domain: str, difficulty: Difficulty) -> list[dict]:
        try:
            return await self.kb_retriever.grounding_examples(
                domain, self._kb_difficulty(difficulty), n=3
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grounding fetch failed for %s: %s", domain, exc)
            return []

    async def _insert(self, docs: list) -> int:
        if not docs:
            return 0
        try:
            result = await get_question_bank().insert_many([d.to_mongo() for d in docs])
            return len(result.inserted_ids)
        except PyMongoError as exc:
            raise DBError(f"Failed to insert into question_bank: {exc}") from exc

    def _qb_difficulty_filter(self, difficulty: Difficulty) -> dict:
        if difficulty == Difficulty.MIXED:
            return {}
        return {"difficulty": difficulty.value}

    def _kb_difficulty(self, difficulty: Difficulty) -> str | None:
        if difficulty == Difficulty.MIXED:
            return None
        return difficulty.value.lower()

    def _respond(self, reused: int, inserted: int) -> GenerateQuestionsResponse:
        return GenerateQuestionsResponse(
            success=True,
            message="Questions generated successfully",
            count=compute_response_count(reused, inserted),
        )
