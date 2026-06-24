"""POST /generate-questions — thin route: validate, delegate, return the fixed response shape.

All branching logic lives in services/quiz_generator.py. This file only:
  - rejects DAILY/SESSION with 501 (Section 6.4),
  - delegates to the orchestrator,
  - maps domain exceptions to HTTP status codes and the fixed response shape.

Pydantic handles enum/range validation (-> 422) before this handler runs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.errors import DBError, LLMError
from app.models.enums import QuizType
from app.models.request import GenerateQuestionsRequest
from app.models.response import GenerateQuestionsResponse
from app.services.quiz_generator import QuizGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

_generator: QuizGenerator | None = None


def get_quiz_generator() -> QuizGenerator:
    global _generator
    if _generator is None:
        _generator = QuizGenerator()
    return _generator


def _fail(status: int, message: str) -> JSONResponse:
    body = GenerateQuestionsResponse(success=False, message=message, count=0)
    return JSONResponse(status_code=status, content=body.model_dump())


@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    # DAILY / SESSION are accepted by the enum but explicitly unimplemented (Section 6.4).
    if request.type in (QuizType.DAILY, QuizType.SESSION):
        return _fail(
            501,
            f"type='{request.type.value}' is not implemented yet. Only CUSTOM is supported.",
        )

    try:
        return await get_quiz_generator().generate_questions(request)
    except LLMError as exc:
        logger.error("LLM failure: %s", exc)
        return _fail(502, f"Question generation failed (LLM error): {exc}")
    except DBError as exc:
        logger.error("DB failure: %s", exc)
        return _fail(503, f"Question generation failed (database error): {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure in /generate-questions")
        return _fail(500, f"Unexpected error: {exc}")
