"""Shared enums for the API contract and data model.

Note the two difficulty vocabularies in this codebase:
- The API request and `question_bank` use UPPERCASE (BEGINNER/INTERMEDIATE/ADVANCED/MIXED).
- `knowledge_base` internally stores lowercase (beginner/intermediate/advanced) and has no
  "mixed" — MIXED is a request-level concept resolved across the three concrete levels.
Mapping between them lives in app/utils/transforms.py, nowhere else.
"""

from __future__ import annotations

from enum import Enum


class QuizType(str, Enum):
    CUSTOM = "CUSTOM"
    DAILY = "DAILY"
    SESSION = "SESSION"


class Difficulty(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    MIXED = "MIXED"


# Concrete difficulty levels that actually exist in knowledge_base (MIXED is not one of them).
CONCRETE_DIFFICULTIES: tuple[Difficulty, ...] = (
    Difficulty.BEGINNER,
    Difficulty.INTERMEDIATE,
    Difficulty.ADVANCED,
)
