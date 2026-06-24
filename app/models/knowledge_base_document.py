"""knowledge_base document schema — Section 4.1 (internal seed content).

Richer than question_bank and never exposed to the frontend. The option/answer shape
matches question_bank exactly so copying across is a near-direct field mapping — but note
the snake_case `correct_answer` here vs camelCase `correctAnswer` in question_bank.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.question_bank_document import Option


class KnowledgeBaseDocument(BaseModel):
    domain: str
    topic: str
    subtopics: list[str] = Field(default_factory=list)
    difficulty: str  # lowercase: beginner / intermediate / advanced
    question: str
    options: list[Option]
    correct_answer: list[str]  # snake_case — internal vocabulary
    explanation: str
    job_relevance: str = ""
    # Content embedding (question + explanation) for grounding-context retrieval (Section 9.2).
    embedding: list[float] = Field(default_factory=list)
    # Topic-matching embedding (topic + subtopics text) for Section 8.1.
    topic_embedding: list[float] = Field(default_factory=list)
    quality_reviewed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        return self.model_dump()
