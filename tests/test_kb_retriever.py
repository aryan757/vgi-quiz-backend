"""Tests for KnowledgeBaseRetriever and utils/transforms.py (domain-level schema)."""

from __future__ import annotations

import pytest

from app.models.enums import Difficulty
from app.services.kb_retriever import KnowledgeBaseRetriever
from app.utils.transforms import (
    difficulty_kb_to_qb,
    difficulty_request_to_kb,
    knowledge_base_to_question_bank,
)

SAMPLE_KB_DOC = {
    "_id": "abc123",
    "domain": "genai",
    "difficulty": "intermediate",
    "question": "What is the primary advantage of the Transformer over RNNs?",
    "options": [
        {"key": "A", "value": "Lower memory usage", "position": 1},
        {"key": "B", "value": "Parallel processing of sequences", "position": 2},
        {"key": "C", "value": "Smaller model size", "position": 3},
        {"key": "D", "value": "No need for training data", "position": 4},
    ],
    "correct_answer": ["B"],
    "explanation": "Transformers process tokens in parallel using self-attention.",
    "job_relevance": "frequently asked",
    "embedding": [0.1, 0.2, 0.3],
    "domain_embedding": [0.4, 0.5, 0.6],
    "quality_reviewed": False,
}


# ---------------------------------------------------------------------------
# transforms.py
# ---------------------------------------------------------------------------

def test_difficulty_kb_to_qb():
    assert difficulty_kb_to_qb("intermediate") == "INTERMEDIATE"
    assert difficulty_kb_to_qb("beginner") == "BEGINNER"
    assert difficulty_kb_to_qb("mixed") == "MIXED"


def test_difficulty_request_to_kb():
    assert difficulty_request_to_kb(Difficulty.BEGINNER) == "beginner"
    assert difficulty_request_to_kb(Difficulty.ADVANCED) == "advanced"


def test_difficulty_mixed_returns_none():
    assert difficulty_request_to_kb(Difficulty.MIXED) is None


def test_knowledge_base_to_question_bank_field_mapping():
    doc = knowledge_base_to_question_bank(SAMPLE_KB_DOC)
    assert hasattr(doc, "correctAnswer")
    assert doc.correctAnswer == ["B"]
    assert doc.difficulty == "INTERMEDIATE"
    assert doc.topic == "genai"        # domain stored in topic field
    assert len(doc.options) == 4


def test_to_mongo_no_internal_fields():
    doc = knowledge_base_to_question_bank(SAMPLE_KB_DOC)
    mongo_dict = doc.to_mongo()
    assert "correctAnswer" in mongo_dict
    assert "correct_answer" not in mongo_dict
    assert "embedding" not in mongo_dict
    assert "domain" not in mongo_dict


# ---------------------------------------------------------------------------
# KnowledgeBaseRetriever (mocked Mongo)
# ---------------------------------------------------------------------------

class MockAsyncCursor:
    def __init__(self, docs): self._docs = docs; self._idx = 0
    def limit(self, n): self._docs = self._docs[:n]; return self
    def __aiter__(self): return self
    async def __anext__(self):
        if self._idx >= len(self._docs): raise StopAsyncIteration
        doc = self._docs[self._idx]; self._idx += 1; return doc


class MockCollection:
    def __init__(self, docs): self._docs = docs
    def find(self, query, *args, **kwargs):
        filtered = [d for d in self._docs if all(d.get(k) == v for k, v in query.items())]
        return MockAsyncCursor(filtered)


class MockEmbeddings:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_fetch_returns_docs(monkeypatch):
    monkeypatch.setattr("app.services.kb_retriever.get_knowledge_base",
                        lambda: MockCollection([SAMPLE_KB_DOC]))
    retriever = KnowledgeBaseRetriever(embeddings=MockEmbeddings())
    result = await retriever.fetch("genai", "intermediate", 5, rank=False)
    assert len(result) == 1
    assert result[0]["domain"] == "genai"


@pytest.mark.asyncio
async def test_fetch_zero_limit(monkeypatch):
    monkeypatch.setattr("app.services.kb_retriever.get_knowledge_base",
                        lambda: MockCollection([SAMPLE_KB_DOC]))
    retriever = KnowledgeBaseRetriever(embeddings=MockEmbeddings())
    assert await retriever.fetch("genai", "intermediate", 0) == []


@pytest.mark.asyncio
async def test_fetch_no_matching_docs(monkeypatch):
    monkeypatch.setattr("app.services.kb_retriever.get_knowledge_base",
                        lambda: MockCollection([SAMPLE_KB_DOC]))
    retriever = KnowledgeBaseRetriever(embeddings=MockEmbeddings())
    result = await retriever.fetch("computer_vision", "beginner", 5)
    assert result == []
