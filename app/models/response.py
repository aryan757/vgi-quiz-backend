"""GenerateQuestionsResponse — Section 6.3 of the spec.

Always this exact shape, on success and failure alike. On failure, `success=False` and
`message` describes the problem; the route sets the appropriate non-200 HTTP status.
"""

from __future__ import annotations

from pydantic import BaseModel


class GenerateQuestionsResponse(BaseModel):
    success: bool
    message: str
    count: int
