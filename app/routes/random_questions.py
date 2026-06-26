"""GET /random-questions — pull N random questions mixed across ALL domains.

Uses MongoDB's $sample aggregation to fetch a uniform random set straight from the
knowledge_base, regardless of domain or difficulty. Embeddings are excluded from the
output. This is a read-only convenience endpoint; it never calls the LLM.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.db import get_knowledge_base

logger = logging.getLogger(__name__)
router = APIRouter()

_PROJECTION = {
    "_id": 0,
    "domain": 1,
    "difficulty": 1,
    "question": 1,
    "options": 1,
    "correct_answer": 1,
    "explanation": 1,
    "job_relevance": 1,
}


@router.get("/random-questions")
async def random_questions(count: int = Query(15, ge=1, le=50)):
    """Return `count` (default 15) random questions mixed from every domain."""
    try:
        kb = get_knowledge_base()
        pipeline = [
            {"$sample": {"size": count}},
            {"$project": _PROJECTION},
        ]
        questions = [doc async for doc in kb.aggregate(pipeline)]
        return {
            "success": True,
            "message": f"Fetched {len(questions)} random questions across all domains.",
            "count": len(questions),
            "questions": questions,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch random questions")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": f"Failed to fetch random questions: {exc}",
                "count": 0,
                "questions": [],
            },
        )
