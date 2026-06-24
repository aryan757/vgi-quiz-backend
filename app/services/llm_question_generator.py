"""Step 4 — full LLM generation (Section 9.4).

Builds a LangChain structured-output chain constrained directly to the question_bank shape
(no regex parsing of free text), injects grounding examples from knowledge_base, and retries
once with an error-correction prompt on validation failure before giving up.

Partial results are allowed: if the model returns fewer valid questions than requested after
the retry, we return what we have and let the orchestrator report the actual count
(Section 6.4 — don't hard-fail a partial result).
"""

from __future__ import annotations

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.models.enums import CONCRETE_DIFFICULTIES, Difficulty
from app.models.question_bank_document import Option, QuestionBankDocument
from app.services.llm_provider import get_chat_model

logger = logging.getLogger(__name__)


# --- structured output schema (constrained to question_bank fields) -------------

class GeneratedOption(BaseModel):
    key: str = Field(description="Option label: A, B, C, or D")
    value: str = Field(description="The option text")
    position: int = Field(description="1-based position, matching the key order")


class GeneratedQuestion(BaseModel):
    question: str
    options: list[GeneratedOption]
    correctAnswer: list[str] = Field(description="List of correct option keys, e.g. ['B']")
    explanation: str
    difficulty: str | None = Field(
        default=None,
        description="BEGINNER, INTERMEDIATE, or ADVANCED. Only used when a MIXED set is requested.",
    )


class GeneratedQuestionSet(BaseModel):
    questions: list[GeneratedQuestion]


_SYSTEM = (
    "You are an expert technical interviewer creating multiple-choice questions (MCQs) to "
    "assess job-readiness for AI/ML engineering roles. Produce clear, unambiguous questions "
    "calibrated to real job interviews — NOT research-level math or proofs. Every question "
    "must have exactly 4 options (keys A, B, C, D with positions 1-4), exactly one correct "
    "answer unless the question explicitly calls for multiple, and a concise explanation of "
    "why the answer is correct."
)


class LLMQuestionGenerator:
    def __init__(self, chat_model=None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chat_model = chat_model

    async def generate(
        self,
        *,
        topic: str,
        difficulty: Difficulty,
        count: int,
        description: str | None = None,
        job_description: str | None = None,
        grounding_examples: list[dict] | None = None,
        quiz_type: str = "CUSTOM",
    ) -> list[QuestionBankDocument]:
        if count <= 0:
            return []

        prompt = self._build_prompt(
            topic=topic, difficulty=difficulty, count=count,
            description=description, job_description=job_description,
            grounding_examples=grounding_examples or [],
        )
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]

        result = await self._invoke(messages)
        if result is None:
            # Retry once with an error-correction nudge (OutputFixingParser pattern).
            logger.warning("LLM structured output failed; retrying once with correction prompt.")
            messages.append(HumanMessage(content=(
                "Your previous response was not valid structured output. Return ONLY a JSON "
                "object with a top-level 'questions' array. Each question must have: question "
                "(string), options (array of {key, value, position}), correctAnswer (array of "
                "strings), explanation (string)."
            )))
            result = await self._invoke(messages)

        if result is None or not result.questions:
            logger.error("LLM generation produced no valid questions for topic=%s.", topic)
            return []

        docs = self._to_documents(result.questions, topic, difficulty, quiz_type)
        if len(docs) < count:
            logger.warning(
                "LLM partial result for topic=%s: asked %d, got %d valid.", topic, count, len(docs)
            )
        return docs[:count]

    # --- internals ----------------------------------------------------------

    def _model(self):
        if self._chat_model is None:
            self._chat_model = get_chat_model(self._settings)
        return self._chat_model

    async def _invoke(self, messages) -> GeneratedQuestionSet | None:
        try:
            structured = self._model().with_structured_output(GeneratedQuestionSet)
            return await structured.ainvoke(messages)
        except (OutputParserException, ValueError, TypeError) as exc:
            logger.warning("Structured output parse error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001  (network/provider errors bubble as None -> retry)
            logger.warning("LLM invocation error: %s", exc)
            return None

    def _build_prompt(
        self, *, topic, difficulty, count, description, job_description, grounding_examples
    ) -> str:
        lines = [f"Generate {count} job-interview-calibrated MCQs on the topic: \"{topic}\"."]

        if difficulty == Difficulty.MIXED:
            lines.append(
                "Use a MIXED difficulty spread across BEGINNER, INTERMEDIATE and ADVANCED, and "
                "set each question's 'difficulty' field accordingly."
            )
        else:
            lines.append(
                f"All questions must be at {difficulty.value} difficulty. "
                f"Set each question's 'difficulty' field to {difficulty.value}."
            )

        if description:
            lines.append(f"Additional focus/context: {description}")
        if job_description:
            lines.append(f"Calibrate relevance toward this job description: {job_description}")

        if grounding_examples:
            lines.append(
                "\nHere are representative example questions for style and accuracy reference "
                "(do NOT copy them verbatim — produce new questions):"
            )
            for i, ex in enumerate(grounding_examples, 1):
                opts = "; ".join(f"{o['key']}) {o['value']}" for o in ex.get("options", []))
                lines.append(
                    f"  Example {i}: {ex.get('question', '')} [{opts}] "
                    f"Answer: {ex.get('correct_answer')}"
                )

        lines.append(
            "\nReturn exactly the requested number of questions. No research-level math. "
            "Each question must have exactly 4 options."
        )
        return "\n".join(lines)

    def _to_documents(
        self, questions: list[GeneratedQuestion], topic: str, difficulty: Difficulty, quiz_type: str
    ) -> list[QuestionBankDocument]:
        valid_levels = {d.value for d in CONCRETE_DIFFICULTIES}
        docs: list[QuestionBankDocument] = []
        for q in questions:
            if not q.options or not q.correctAnswer or not q.question.strip():
                continue  # skip malformed; partial result is acceptable

            if difficulty == Difficulty.MIXED:
                level = (q.difficulty or "").strip().upper()
                doc_difficulty = level if level in valid_levels else Difficulty.INTERMEDIATE.value
            else:
                doc_difficulty = difficulty.value  # stamp requested level, don't trust model

            try:
                docs.append(QuestionBankDocument(
                    type=quiz_type,
                    topic=topic,
                    difficulty=doc_difficulty,
                    question=q.question.strip(),
                    options=[Option(key=o.key, value=o.value, position=o.position) for o in q.options],
                    correctAnswer=list(q.correctAnswer),
                    explanation=q.explanation.strip(),
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dropping a malformed generated question: %s", exc)
        return docs
