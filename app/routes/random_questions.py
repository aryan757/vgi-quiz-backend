"""GET /random-questions — AI-generate N questions covering ALL domains.

Each call generates a fresh set of MCQs with the LLM (NOT read from knowledge_base),
spread across all domains at an easy-medium level. The generated batch is saved to the
`random-question-collection` Mongo collection and returned to the caller.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.db import get_random_question_collection
from app.services.random_question_generator import RandomQuestionGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

_generator: RandomQuestionGenerator | None = None


def get_random_generator() -> RandomQuestionGenerator:
    global _generator
    if _generator is None:
        _generator = RandomQuestionGenerator()
    return _generator


@router.get("/random-questions")
async def random_questions(count: int = Query(15, ge=1, le=50)):
    """AI-generate `count` (default 15) easy-medium questions across all domains."""
    try:
        questions = await get_random_generator().generate(count)
        if not questions:
            return JSONResponse(
                status_code=502,
                content={
                    "success": False,
                    "message": "Question generation failed (LLM returned no valid questions).",
                    "count": 0,
                    "questions": [],
                },
            )

        # Save the generated batch to its own collection (a log of what was produced).
        batch_id = str(uuid.uuid4())
        try:
            await get_random_question_collection().insert_one({
                "batch_id": batch_id,
                "count": len(questions),
                "questions": questions,
                "created_at": datetime.now(timezone.utc),
            })
        except Exception as exc:  # noqa: BLE001 — saving must not break the response
            logger.warning("Failed to save random-question batch %s: %s", batch_id, exc)

        return {
            "success": True,
            "message": f"Generated {len(questions)} questions across all domains.",
            "count": len(questions),
            "questions": questions,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate random questions")
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "message": f"Failed to generate random questions: {exc}",
                "count": 0,
                "questions": [],
            },
        )
