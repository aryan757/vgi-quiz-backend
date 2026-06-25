"""Tests for TopicMatcher (domain matching).

Uses a mocked EmbeddingsClient — no live OpenAI calls needed.
"""

from __future__ import annotations

import pytest

from app.services.topic_matcher import MatchResult, TopicMatcher


class MockEmbeddings:
    def __init__(self, registry: dict | None = None):
        self.registry = registry or {}

    async def embed(self, text: str) -> list[float]:
        return self.registry.get(text.lower(), [1.0, 0.0, 0.0])

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


def make_matcher(**kw):
    return TopicMatcher(embeddings=MockEmbeddings(), **kw)


# ---------------------------------------------------------------------------
# auto topic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_topic_returns_no_match():
    matcher = make_matcher()
    result = await matcher.match("auto")
    assert result.is_auto is True
    assert result.matched is False
    assert result.method == "auto"


@pytest.mark.asyncio
async def test_auto_topic_mixed_case():
    result = await make_matcher().match("AUTO")
    assert result.is_auto is True


# ---------------------------------------------------------------------------
# Fast path — domain name / alias matching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exact_domain_name():
    result = await make_matcher().match("computer_vision")
    assert result.matched is True
    assert result.canonical_domain == "computer_vision"
    assert result.method == "fast_path"


@pytest.mark.asyncio
async def test_alias_cv():
    result = await make_matcher().match("cv")
    assert result.matched is True
    assert result.canonical_domain == "computer_vision"


@pytest.mark.asyncio
async def test_alias_llm():
    result = await make_matcher().match("llm")
    assert result.matched is True
    assert result.canonical_domain == "genai"


@pytest.mark.asyncio
async def test_alias_neural_network():
    result = await make_matcher().match("neural network")
    assert result.matched is True
    assert result.canonical_domain == "deep_learning"


@pytest.mark.asyncio
async def test_alias_partial_containment():
    # "yolo" is an alias for computer_vision
    result = await make_matcher().match("yolo detection")
    assert result.matched is True
    assert result.canonical_domain == "computer_vision"


@pytest.mark.asyncio
async def test_unknown_topic_no_kb_candidates():
    matcher = make_matcher()
    matcher._kb_domain_candidates = []
    result = await matcher.match("quantum computing")
    assert result.matched is False
    assert result.method == "none"


# ---------------------------------------------------------------------------
# Embedding fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_match_above_threshold():
    query = "seq2seq positional encoding"
    registry = {query.lower(): [0.95, 0.31, 0.0]}
    matcher = TopicMatcher(embeddings=MockEmbeddings(registry))
    matcher._kb_domain_candidates = [("deep_learning", [0.95, 0.31, 0.0])]
    result = await matcher.match(query)
    assert result.matched is True
    assert result.canonical_domain == "deep_learning"
    assert result.method == "embedding"


@pytest.mark.asyncio
async def test_embedding_match_below_threshold():
    query = "xyzzy frobulation zorp"
    registry = {query.lower(): [0.0, 0.0, 1.0]}
    matcher = TopicMatcher(embeddings=MockEmbeddings(registry))
    matcher._kb_domain_candidates = [("genai", [1.0, 0.0, 0.0])]
    result = await matcher.match(query)
    assert result.matched is False
    assert result.method == "none"


# ---------------------------------------------------------------------------
# Domain inference (for auto topic)
# ---------------------------------------------------------------------------

def test_infer_domain_genai():
    matcher = make_matcher()
    domain = matcher._infer_domain_from_text("LangChain RAG pipeline", None)
    assert domain == "genai"


def test_infer_domain_computer_vision():
    matcher = make_matcher()
    domain = matcher._infer_domain_from_text(None, "Looking for a YOLO detection engineer")
    assert domain == "computer_vision"


def test_infer_domain_none_when_no_text():
    assert make_matcher()._infer_domain_from_text(None, None) is None
