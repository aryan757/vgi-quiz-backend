"""Tests for the orchestration flow — all branch paths.

Branch A (full reuse)   : domain match, question_bank already has >= requested.
Branch A (shortfall)    : domain match, partial inventory → top up from KB.
Branch B (KB seed)      : domain match, zero inventory → pull from KB.
Branch C (LLM only)     : description/job_description present OR no domain match.
"""

from __future__ import annotations

import pytest

from app.models.enums import Difficulty, QuizType
from app.models.question_bank_document import Option, QuestionBankDocument
from app.models.request import GenerateQuestionsRequest
from app.services.quiz_generator import QuizGenerator
from app.services.topic_matcher import MatchResult


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

def make_qb_doc(domain="genai", difficulty="INTERMEDIATE") -> QuestionBankDocument:
    return QuestionBankDocument(
        type="CUSTOM", topic=domain, difficulty=difficulty,
        question="Sample question?",
        options=[
            Option(key="A", value="opt A", position=1),
            Option(key="B", value="opt B", position=2),
            Option(key="C", value="opt C", position=3),
            Option(key="D", value="opt D", position=4),
        ],
        correctAnswer=["A"], explanation="Because A.",
    )


SAMPLE_KB_RAW = {
    "_id": "kb1",
    "domain": "genai",
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

    def _infer_domain_from_text(self, desc, jd) -> str | None:
        return None


class MockKBRetriever:
    def __init__(self, docs: list[dict]):
        self._docs = docs

    async def fetch(self, domain, difficulty, limit, **kwargs) -> list[dict]:
        return self._docs[:limit]

    async def grounding_examples(self, domain, difficulty, n=3, **kwargs) -> list[dict]:
        return self._docs[:n]


class MockLLMGenerator:
    def __init__(self, docs: list[QuestionBankDocument]):
        self._docs = docs

    async def generate(self, *, count, **kwargs) -> list[QuestionBankDocument]:
        return self._docs[:count]


class MockQuestionBank:
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


def make_generator(match_result, kb_docs, llm_docs, existing_count=0):
    mock_qb = MockQuestionBank(existing_count=existing_count)
    gen = QuizGenerator(
        topic_matcher=MockTopicMatcher(match_result),
        kb_retriever=MockKBRetriever(kb_docs),
        llm_generator=MockLLMGenerator(llm_docs),
    )
    return gen, mock_qb


CONFIDENT_MATCH = MatchResult(matched=True, canonical_domain="genai", score=0.95, method="fast_path")
NO_MATCH = MatchResult(matched=False, canonical_domain=None, score=0.3, method="none")


# ---------------------------------------------------------------------------
# Branch A — full reuse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_a_full_reuse(monkeypatch):
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], [], existing_count=10)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="genai", question_count=10)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 10
    assert len(mock_qb.inserted) == 0


@pytest.mark.asyncio
async def test_branch_a_full_reuse_caps_at_requested(monkeypatch):
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], [], existing_count=50)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="genai", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.count == 5
    assert len(mock_qb.inserted) == 0


# ---------------------------------------------------------------------------
# Branch A — shortfall
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_a_shortfall_from_kb(monkeypatch):
    kb_docs = [SAMPLE_KB_RAW.copy(), SAMPLE_KB_RAW.copy()]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, kb_docs, [], existing_count=3)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="genai", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 2


# ---------------------------------------------------------------------------
# Branch B — zero inventory, full KB seed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_b_full_kb_seed(monkeypatch):
    kb_docs = [SAMPLE_KB_RAW.copy() for _ in range(5)]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, kb_docs, [], existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="genai", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 5


@pytest.mark.asyncio
async def test_branch_b_kb_short_tops_up_via_llm(monkeypatch):
    kb_docs = [SAMPLE_KB_RAW.copy(), SAMPLE_KB_RAW.copy()]
    llm_docs = [make_qb_doc() for _ in range(3)]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, kb_docs, llm_docs, existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="genai", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 5


# ---------------------------------------------------------------------------
# Branch C — full LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch_c_no_domain_match(monkeypatch):
    llm_docs = [make_qb_doc() for _ in range(5)]
    gen, mock_qb = make_generator(NO_MATCH, [], llm_docs, existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(type=QuizType.CUSTOM, topic="quantum computing", question_count=5)
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 5
    assert len(mock_qb.inserted) == 5


@pytest.mark.asyncio
async def test_branch_c_context_bypasses_domain_match(monkeypatch):
    llm_docs = [make_qb_doc() for _ in range(3)]
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], llm_docs, existing_count=0)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(
        type=QuizType.CUSTOM, topic="genai",
        description="Focus on cross-attention in diffusion models",
        question_count=3,
    )
    resp = await gen.generate_questions(req)

    assert resp.success is True
    assert resp.count == 3
    assert len(mock_qb.inserted) == 3


# ---------------------------------------------------------------------------
# MIXED difficulty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_difficulty_no_filter(monkeypatch):
    gen, mock_qb = make_generator(CONFIDENT_MATCH, [], [], existing_count=10)
    monkeypatch.setattr("app.services.quiz_generator.get_question_bank", lambda: mock_qb)

    req = GenerateQuestionsRequest(
        type=QuizType.CUSTOM, topic="genai", difficulty=Difficulty.MIXED, question_count=10
    )
    resp = await gen.generate_questions(req)
    assert resp.count == 10
