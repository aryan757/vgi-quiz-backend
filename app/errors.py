"""Domain exceptions, mapped to HTTP status codes in the route (Section 6.4)."""

from __future__ import annotations


class QuizGenerationError(Exception):
    """Base class for generation-flow failures."""


class LLMError(QuizGenerationError):
    """LLM call failed (timeout/API error) after retry -> HTTP 502."""


class DBError(QuizGenerationError):
    """Mongo connection/write failure -> HTTP 503."""


class NotImplementedQuizType(QuizGenerationError):
    """DAILY / SESSION requested -> HTTP 501."""
