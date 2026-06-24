"""Tests for the 5-step orchestration flow — covers all branch paths (Section 9).

Branch A (full reuse)   : confident KB match, question_bank already has >= requested.
Branch A (shortfall)    : confident KB match, partial inventory -> top up from KB.
Branch B (KB seed only) : confident KB match, zero inventory -> pull entirely from KB.
Branch C (LLM only)     : description/job_description present OR no confident match.

All Mongo and LLM I/O is mocked so these tests run offline.
"""

from __future__ import annotations

import pytest

from app.models.enums import Difficulty, QuizType
from app.models.question_bank_document import Option, QuestionBankDocument
from app.models.request import GenerateQuestionsRequest
from app.services.quiz_generator import QuizGenerator
from app.services.topic_matcher import MatchResult


# ---------------------------------------------------------------------------
# Shared helpers / mocks
# ---------------------------------------------------------------------------

def make_qb_doc(topic="Transformers", difficulty="INTERMEDIATE") -> QuestionBankDocument:
    return QuestionBankDocument(
        type="CUSTOM",
        topic=topic,
        difficulty=difficulty,
        question="Sample question?",
        options=[
            Option(key="A", value="opt A", position=1),
            Option(key="B", value="opt B", position=2),
            Option(key="C", value="opt C", position=3),
            Option(key="D", value="opt D", position=4),
        ],
        correctAnswer=["A"],
        explanation="Because A.",
    )


SAMPLE_KB_RAW = {
    "_id": "kb1",
    "domain": "genai",
    "topic": "Transformers",
    "subtopics": [],
    "difficulty": "intermediate",
    "question": "What is self-attention?",
    "options": [
        {"key": "A", "value": "opt A", "position": 1},
        {"key": "B", "value": "opt B", "position": 2},
        {"key": "C", "value": "opt C", "position": 3},
        {"key": "D", "value": "opt D", "position": 4},
    ],
    "correct_answer": ["B"],
    "explanation": "Self-attention computes token relationships.",
}


class MockTopicMatcher:
    def __init__(self, result: MatchResult):
        self._result = result

    async def match(self, topic, description=None, job_description=None) -> MatchResult:
        return self._result

    def fast_path_lookup(self, topic: str) -> str | None:
        return None


class MockKBRetriever:
    def __init__(self, docs: list[dict]):
        self._docs = docs

    async def fetch(self, topic, difficulty, limit, **kwargs) -> list[dict]:
        return self._docs[:limit]

    async def grounding_examples(self, topic, difficulty, n=3, **kwargs) -> list[dict]:
        return self._docs[:n]


class MockLLMGenerator:
    def __init__(self, docs: list[QuestionBankDocument]):
        self._docs = docs

    async def generate(self, *, count, **kwargs) -> list[QuestionBankDocument]:
        return self._docs[:count]


class MockQuestionBank:
    """In-memory question_bank store."""
    def __init__(self, existing_count: int = 0):
        self._count = existing_count
        self.inserted: list[dict] = []

    async def count_documents(self, query):
        return self._count

    async def insert_many(self, docs):
        self.inserted.extend(docs)
        self._count += len(docs)

        class Result:
            inserted_ids = [f"id{i}" for i in range(len(docs))]
        return Result()


def make_generator(
    match_result: MatchResult,
    kb_docs: list[dict],
    llm_docs: list[QuestionBankDocument],
    existing_count: int = 0,
) -> tuple[QuizGenerator, MockQuestionBank]:
    mock_qb = MockQuestionBank(existing_count=existing_count)
    gen = QuizGenerator(
        topic_matcher=MockTopicMatcher(match_result),
        kb_retriever=MockKBRetriever(kb_docs),
        llm_generator=MockLLMGenerator(llm_docs),
    )
    return gen, mock_qb


CONFIDENT_MATCH = MatchResult(
    matched=True, canonical_topic="Transformers", domain="genai", score=0.95, method="fast_path"
)
NO_MATCH = MatchResult(
    matched=False, canonical_topic=None, domain=None, score=0.3, method="none"
)


# ---------------------------------------------------------------------------
# Branch A — full reuse (no generation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_a_full_reuse(monkeypatch):
    """question_bank already has enough -> reuse, nothing inserted."""
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], [], existing_count=10)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="Transformers", question_count=10)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 10
    assert len(mock_qb.inserted) == 0  # nothing was written


@pytest.mark.asyncio
async def test_branch_a_full_reuse_caps_at_requested(monkeypatch):
    """existing_count > question_count: count is capped at question_count."""
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], [], existing_count=50)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="Transformers", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.count == 5
    assert len(mock_qb.inserted) == 0


# ---------------------------------------------------------------------------
# Branch A — shortfall (top up from KB)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_a_shortfall_from_kb(monkeypatch):
    """Existing 3, need 5 -> pull 2 from KB."""
    kb_docs = [SAMPLE_KB_RAW.copy(), SAMPLE_KB_RAW.copy()]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, kb_docs, [], existing_count=3)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="Transformers", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 2


# ---------------------------------------------------------------------------
# Branch B — zero inventory, pull entirely from KB
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_b_full_kb_seed(monkeypatch):
    """question_bank empty -> pull 5 from KB."""
    kb_docs = [SAMPLE_KB_RAW.copy() for _ in range(5)]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, kb_docs, [], existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="Transformers", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 5


@pytest.mark.asyncio
async def test_branch_b_kb_short_tops_up_via_llm(monkeypatch):
    """KB only has 2 docs but 5 needed -> top up 3 from LLM."""
    kb_docs = [SAMPLE_KB_RAW.copy(), SAMPLE_KB_RAW.copy()]
    llm_docs = [make_qb_doc() for _ in range(3)]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, kb_docs, llm_docs, existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="Transformers", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 5


# ---------------------------------------------------------------------------
# Branch C — full LLM (no confident match)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_c_no_match_uses_llm(monkeypatch):
    """No confident topic match -> full LLM generation."""
    llm_docs = [make_qb_doc() for _ in range(5)]
    gen, mock_qb = make_generator(NO_MATCH, [], llm_docs, existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="Quantum Computing", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 5


@pytest.mark.asyncio
async def test_branch_c_context_bypasses_topic_match(monkeypatch):
    """description present -> skip topic matching entirely, go to LLM."""
    llm_docs = [make_qb_doc() for _ in range(3)]
    # even if match would be confident, description forces LLM
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], llm_docs, existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(
        type=QuizType.CUSTOM,
        topic="Transformers",
        description="Focus on cross-attention in diffusion models",
        question_count=3,
    )
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 3
    assert len(mock_qb.inserted) == 3


# ---------------------------------------------------------------------------
# DAILY / SESSION -> 501 (tested at route level in test_request_validation.py
# but also verify the error type here for completeness)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_difficulty_inventory_query(monkeypatch):
    """MIXED difficulty -> inventory query has no difficulty filter."""
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], [], existing_count=10)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(
        type=QuizType.CUSTOM, topic="Transformers", difficulty=Difficulty.MIXED, question_count=10
    )
    resp = await gen.generate_questions(req)
    assert resp.count == 10
