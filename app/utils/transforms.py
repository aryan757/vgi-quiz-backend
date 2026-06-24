"""Field mapping between the two collections — the ONLY place that knows the differences.

knowledge_base (internal)        question_bank (frontend-facing)
  correct_answer  (snake)   ->     correctAnswer   (camel)
  difficulty: lowercase     ->     difficulty: UPPERCASE
  created_at                 ->     createdAt / updatedAt (fresh on copy)

Also handles the difficulty vocabulary mapping in both directions. Do not duplicate any of
this logic elsewhere (Section 12).
"""

from __future__ import annotations

from app.models.enums import Difficulty
from app.models.question_bank_document import Option, QuestionBankDocument


def difficulty_kb_to_qb(kb_difficulty: str) -> str:
    """lowercase KB difficulty -> UPPERCASE question_bank difficulty."""
    return kb_difficulty.strip().upper()


def difficulty_request_to_kb(difficulty: Difficulty) -> str:
    """UPPERCASE request difficulty -> lowercase KB difficulty.

    MIXED has no single KB level; callers must expand MIXED across CONCRETE_DIFFICULTIES
    before reaching here.
    """
    if difficulty == Difficulty.MIXED:
        raise ValueError("MIXED must be expanded to a concrete difficulty before KB mapping.")
    return difficulty.value.lower()


def knowledge_base_to_question_bank(
    kb_doc: dict,
    *,
    quiz_type: str = "CUSTOM",
    topic_override: str | None = None,
) -> QuestionBankDocument:
    """Map a knowledge_base document (dict) into a fresh question_bank document.

    `topic_override` lets the caller stamp the matched canonical topic name (e.g. when the
    request topic differed in casing/wording from the stored KB topic).
    """
    options = [
        Option(key=o["key"], value=o["value"], position=o["position"])
        for o in kb_doc["options"]
    ]
    return QuestionBankDocument(
        type=quiz_type,
        topic=topic_override or kb_doc["topic"],
        difficulty=difficulty_kb_to_qb(kb_doc["difficulty"]),
        question=kb_doc["question"],
        options=options,
        correctAnswer=list(kb_doc["correct_answer"]),  # snake -> camel, here and nowhere else
        explanation=kb_doc.get("explanation", ""),
    )
