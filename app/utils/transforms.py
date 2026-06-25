"""Field mapping between knowledge_base and question_bank — the ONLY place that knows
the differences between the two schemas.

knowledge_base (internal)          question_bank (frontend-facing)
  correct_answer  (snake)    →       correctAnswer  (camel)
  difficulty: lowercase      →       difficulty: UPPERCASE
  domain                     →       topic (reused as the domain name)
  no createdAt/updatedAt     →       createdAt / updatedAt (fresh on copy)
"""

from __future__ import annotations

from app.models.enums import Difficulty
from app.models.question_bank_document import Option, QuestionBankDocument


def difficulty_kb_to_qb(kb_difficulty: str) -> str:
    """lowercase/mixed KB difficulty → UPPERCASE question_bank difficulty."""
    return kb_difficulty.strip().upper()


def difficulty_request_to_kb(difficulty: Difficulty) -> str | None:
    """UPPERCASE request difficulty → lowercase KB difficulty.
    Returns None for MIXED (no difficulty constraint on KB query).
    """
    if difficulty == Difficulty.MIXED:
        return None
    return difficulty.value.lower()


def knowledge_base_to_question_bank(
    kb_doc: dict,
    *,
    quiz_type: str = "CUSTOM",
) -> QuestionBankDocument:
    """Map a knowledge_base document (dict) into a fresh question_bank document.

    The domain name is stored in question_bank's `topic` field so the frontend
    contract stays unchanged.
    """
    options = [
        Option(key=o["key"], value=o["value"], position=o["position"])
        for o in kb_doc["options"]
    ]
    return QuestionBankDocument(
        type=quiz_type,
        topic=kb_doc["domain"],                         # domain → topic field
        difficulty=difficulty_kb_to_qb(kb_doc["difficulty"]),
        question=kb_doc["question"],
        options=options,
        correctAnswer=list(kb_doc["correct_answer"]),   # snake → camel, here only
        explanation=kb_doc.get("explanation", ""),
    )
