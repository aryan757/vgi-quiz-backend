"""GenerateQuestionsRequest — Section 6.2 of the spec."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models.enums import Difficulty, QuizType


class GenerateQuestionsRequest(BaseModel):
    type: QuizType = Field(..., description="CUSTOM, DAILY, or SESSION. Only CUSTOM is implemented.")
    topic: str = Field(..., min_length=1, description='A topic name, or the literal string "auto".')
    # Defaults to MIXED when omitted (documented default per Section 6.2).
    difficulty: Difficulty = Field(
        default=Difficulty.MIXED,
        description="BEGINNER/INTERMEDIATE/ADVANCED/MIXED. Defaults to MIXED if omitted.",
    )
    description: str | None = Field(default=None, description="Extra generation context for the LLM.")
    job_description: str | None = Field(
        default=None, description="Job description to calibrate relevance toward."
    )
    question_count: int = Field(
        ..., ge=1, le=100, description="How many questions to satisfy. Validated 1..100 -> 422 otherwise."
    )

    @field_validator("topic")
    @classmethod
    def _strip_topic(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic must not be blank")
        return v

    @property
    def is_auto_topic(self) -> bool:
        return self.topic.strip().lower() == "auto"
