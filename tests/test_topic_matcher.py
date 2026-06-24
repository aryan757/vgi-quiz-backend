"""Tests for TopicMatcher (Section 8).

Uses a mocked EmbeddingsClient so no live OpenAI calls are needed.
KB candidate loading is also mocked via monkeypatching.
"""

from __future__ import annotations

import pytest

from app.services.topic_matcher import MatchResult, TopicMatcher


class MockEmbeddings:
    """Returns a deterministic unit vector for any text.
    For similarity testing, supply specific vectors via the `registry` dict.
    """
    def __init__(self, registry: dict | None = None):
        self.registry = registry or {}

    async def embed(self, text: str) -> list[float]:
        return self.registry.get(text.lower(), [1.0, 0.0, 0.0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Fast-path tests (no API calls)
# ---------------------------------------------------------------------------

def make_matcher(**kw):
    return TopicMatcher(embeddings=MockEmbeddings(), **kw)


@pytest.mark.asyncio
async def test_auto_topic_returns_no_match():
    matcher = make_matcher()
    result = await matcher.match("auto")
    assert result.is_auto is True
    assert result.matched is False
    assert result.method == "auto"


@pytest.mark.asyncio
async def test_auto_topic_mixed_case():
    matcher = make_matcher()
    result = await matcher.match("AUTO")
    assert result.is_auto is True


@pytest.mark.asyncio
async def test_exact_match_fast_path():
    matcher = make_matcher()
    result = await matcher.match("Transformers")
    assert result.matched is True
    assert result.canonical_topic == "Transformers"
    assert result.method == "fast_path"
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_case_insensitive_fast_path():
    matcher = make_matcher()
    result = await matcher.match("transformers")
    assert result.matched is True
    assert result.canonical_topic == "Transformers"
    assert result.method == "fast_path"


@pytest.mark.asyncio
async def test_containment_fast_path():
    matcher = make_matcher()
    # "the Transformers guide" contains "transformers" -> matches the "Transformers" canonical topic
    result = await matcher.match("the Transformers guide")
    assert result.matched is True
    assert result.canonical_topic == "Transformers"


@pytest.mark.asyncio
async def test_unknown_topic_no_kb_candidates():
    matcher = make_matcher()
    # Inject empty candidates to skip Mongo and go straight to "no confident match".
    matcher._kb_topic_candidates = []
    result = await matcher.match("Quantum Computing")
    assert result.matched is False
    assert result.method == "none"


# ---------------------------------------------------------------------------
# Embedding fallback tests (mocked KB candidates)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_match_above_threshold(monkeypatch):
    # Use a query with zero lexical overlap with any canonical topic so fast_path
    # definitely doesn't resolve it — then embedding similarity finds the match.
    query = "seq2seq positional encoding"
    registry = {query.lower(): [0.95, 0.31, 0.0]}
    matcher = TopicMatcher(embeddings=MockEmbeddings(registry))
    # Inject KB candidates directly (bypassing Mongo)
    matcher._kb_topic_candidates = [("Transformers", [0.95, 0.31, 0.0])]
    result = await matcher.match(query)
    assert result.matched is True
    assert result.canonical_topic == "Transformers"
    assert result.method == "embedding"


@pytest.mark.asyncio
async def test_embedding_match_below_threshold(monkeypatch):
    # Use a query that has no lexical overlap with any canonical topic, and whose
    # embedding is orthogonal (cosine=0.0) to the only KB candidate -> no confident match.
    query = "xyzzy frobulation zorp"
    registry = {query.lower(): [0.0, 0.0, 1.0]}
    matcher = TopicMatcher(embeddings=MockEmbeddings(registry))
    matcher._kb_topic_candidates = [("RAG", [1.0, 0.0, 0.0])]
    result = await matcher.match(query)
    assert result.matched is False
    assert result.method == "none"


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def test_infer_domain_genai():
    matcher = make_matcher()
    domain = matcher._infer_domain("LangChain RAG pipeline", None)
    assert domain == "genai"


def test_infer_domain_computer_vision():
    matcher = make_matcher()
    domain = matcher._infer_domain(None, "Looking for a YOLO detection engineer")
    assert domain == "computer_vision"


def test_infer_domain_none_when_no_text():
    matcher = make_matcher()
    assert matcher._infer_domain(None, None) is None
