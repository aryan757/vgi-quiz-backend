"""question_bank document schema — Section 4.2 (frontend-facing contract).

DO NOT add fields beyond what's here — the frontend depends on this exact shape.
Note the intentional camelCase fields (`correctAnswer`, `createdAt`, `updatedAt`).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Option(BaseModel):
    key: str
    value: str
    position: int


class QuestionBankDocument(BaseModel):
    type: str = "CUSTOM"
    topic: str
    difficulty: str  # UPPERCASE: BEGINNER / INTERMEDIATE / ADVANCED
    question: str
    options: list[Option]
    correctAnswer: list[str]  # camelCase per the confirmed schema — do NOT "fix"
    explanation: str
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        """Serialize for insertion into question_bank. Lets Mongo assign `_id`."""
        return self.model_dump()
