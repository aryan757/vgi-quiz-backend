"""AI generator for the GET /random-questions endpoint.

Generates a fresh set of MCQs with the LLM (via LangChain structured output) spread
across ALL domains, kept at an easy-medium level. This does NOT read from the
knowledge_base — every call produces new questions and the batch is saved to the
`random-question-collection` Mongo collection by the route.
"""

from __future__ import annotations

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services.llm_provider import get_chat_model

logger = logging.getLogger(__name__)

# The five domains the set must cover.
DOMAINS = ["computer_vision", "machine_learning", "deep_learning", "genai", "ai_fundamentals"]

# Only easy-medium for this endpoint.
ALLOWED_DIFFICULTIES = {"beginner", "intermediate"}


# --- structured output schema -------------------------------------------------

class GenOption(BaseModel):
    key: str = Field(description="Option label: A, B, C, or D")
    value: str = Field(description="The option text")
    position: int = Field(description="1-based position matching the key order (1-4)")


class GenQuestion(BaseModel):
    domain: str = Field(description="One of: " + ", ".join(DOMAINS))
    difficulty: str = Field(description="Either 'beginner' or 'intermediate' (easy-medium only)")
    question: str
    options: list[GenOption]
    correct_answer: list[str] = Field(description="List of correct option keys, e.g. ['B']")
    explanation: str
    job_relevance: str = Field(description="One short line on why this matters for a job")


class GenQuestionSet(BaseModel):
    questions: list[GenQuestion]


_SYSTEM = (
    "You are an expert AI/ML technical interviewer writing multiple-choice questions (MCQs) "
    "to assess job-readiness. Produce clear, unambiguous questions at an EASY-to-MEDIUM level "
    "only — no hard/advanced, no research-level math or proofs. Every question must have exactly "
    "4 options (keys A, B, C, D with positions 1-4), exactly one correct answer, and a concise "
    "explanation. Spread the questions across the requested domains so all of them are covered."
)


def _plan_distribution(count: int) -> dict[str, int]:
    """Spread `count` questions across the domains as evenly as possible."""
    base, extra = divmod(count, len(DOMAINS))
    plan = {d: base for d in DOMAINS}
    for d in DOMAINS[:extra]:  # hand out remainders to the first domains
        plan[d] += 1
    return {d: n for d, n in plan.items() if n > 0}


class RandomQuestionGenerator:
    def __init__(self, chat_model=None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chat_model = chat_model

    async def generate(self, count: int = 15) -> list[dict]:
        if count <= 0:
            return []

        plan = _plan_distribution(count)
        prompt = self._build_prompt(count, plan)
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]

        result = await self._invoke(messages)
        if result is None:
            logger.warning("Random-question structured output failed; retrying once.")
            messages.append(HumanMessage(content=(
                "Your previous response was not valid. Return ONLY a JSON object with a top-level "
                "'questions' array. Each item must have: domain, difficulty, question, options "
                "(array of {key, value, position}), correct_answer (array of strings), explanation, "
                "job_relevance."
            )))
            result = await self._invoke(messages)

        if result is None or not result.questions:
            logger.error("Random-question generation produced nothing.")
            return []

        return self._clean(result.questions)[:count]

    # --- internals ------------------------------------------------------------

    def _model(self):
        if self._chat_model is None:
            self._chat_model = get_chat_model(self._settings, temperature=0.7)
        return self._chat_model

    async def _invoke(self, messages) -> GenQuestionSet | None:
        try:
            structured = self._model().with_structured_output(GenQuestionSet)
            return await structured.ainvoke(messages)
        except (OutputParserException, ValueError, TypeError) as exc:
            logger.warning("Structured output parse error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM invocation error: %s", exc)
            return None

    def _build_prompt(self, count: int, plan: dict[str, int]) -> str:
        spread = "; ".join(f"{n} from {d}" for d, n in plan.items())
        return (
            f"Generate exactly {count} MCQs spread across these domains: {spread}.\n"
            "Set each question's 'domain' field to its domain and 'difficulty' to either "
            "'beginner' or 'intermediate' (easy-medium only). Make every question distinct. "
            "Each question must have exactly 4 options (A, B, C, D with positions 1-4) and "
            "exactly one correct answer. Keep them practical and interview-relevant."
        )

    def _clean(self, questions: list[GenQuestion]) -> list[dict]:
        out: list[dict] = []
        for q in questions:
            if not q.options or not q.correct_answer or not q.question.strip():
                continue
            domain = q.domain.strip().lower()
            if domain not in DOMAINS:
                domain = "ai_fundamentals"
            difficulty = q.difficulty.strip().lower()
            if difficulty not in ALLOWED_DIFFICULTIES:
                difficulty = "intermediate"
            out.append({
                "domain": domain,
                "difficulty": difficulty,
                "question": q.question.strip(),
                "options": [
                    {"key": o.key, "value": o.value, "position": o.position} for o in q.options
                ],
                "correct_answer": list(q.correct_answer),
                "explanation": q.explanation.strip(),
                "job_relevance": q.job_relevance.strip(),
            })
        return out
