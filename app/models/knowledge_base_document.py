"""knowledge_base document schema — domain-level (no topic/subtopics).

Questions are stored per domain + difficulty only. The content covers the full
scope of that domain but individual documents have no topic/subtopic fields.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.question_bank_document import Option


class KnowledgeBaseDocument(BaseModel):
    domain: str                          # computer_vision / machine_learning / deep_learning / genai / ai_fundamentals
    difficulty: str                      # beginner / intermediate / advanced / mixed
    question: str
    options: list[Option]
    correct_answer: list[str]            # snake_case — internal vocabulary
    explanation: str
    job_relevance: str = ""
    embedding: list[float] = Field(default_factory=list)        # content embedding for retrieval ranking
    domain_embedding: list[float] = Field(default_factory=list) # domain name embedding for domain matching
    quality_reviewed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        return self.model_dump()
