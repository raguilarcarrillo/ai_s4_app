"""Quiz schema, generation, and grading for Smart Teacher.

Public entry points:

* :func:`generate_quiz` — produce a structured :class:`Quiz` from a topic and
  optional retrieval context, with parse-retry on failure.
* :func:`grade_quiz` — score user answers against a :class:`Quiz` locally and
  return per-question feedback.

The :class:`Quiz` / :class:`Question` models double as the structured-output
schema for LangChain's ``with_structured_output``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator

from prompts import QUIZ_SYSTEM_PROMPT, build_quiz_user_message
from rag import RetrievedChunk, format_context_block

logger = logging.getLogger(__name__)

QuestionType = Literal["multiple_choice", "true_false", "short_answer"]
Difficulty = Literal["easy", "medium", "hard"]
QuestionTypesArg = Literal[
    "multiple_choice", "true_false", "short_answer", "mixed"
]


class Question(BaseModel):
    """A single quiz question.

    Attributes:
        id: 1-indexed identifier within the quiz.
        type: Question type.
        prompt: The text shown to the learner.
        options: Answer options for ``multiple_choice`` / ``true_false``.
            ``None`` for ``short_answer``.
        correct_answer: Index (int) for MCQ / TF, canonical string for
            short-answer.
        explanation: Why this is the right answer; used during grading.
        source_refs: Chunk ids justifying the answer in grounded mode.
    """

    id: int = Field(..., ge=1)
    type: QuestionType
    prompt: str = Field(..., min_length=1)
    options: Optional[List[str]] = None
    correct_answer: Union[int, str]
    explanation: str = Field(..., min_length=1)
    source_refs: Optional[List[str]] = None

    @field_validator("options")
    @classmethod
    def _validate_options(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Ensure option lists are non-empty when present."""
        if v is not None and len(v) < 2:
            raise ValueError("options must have at least 2 entries when set")
        return v


class Quiz(BaseModel):
    """A collection of :class:`Question` objects for a topic.

    Attributes:
        topic: Subject the quiz covers.
        difficulty: Difficulty calibration.
        questions: Ordered list of questions.
    """

    topic: str = Field(..., min_length=1)
    difficulty: Difficulty
    questions: List[Question] = Field(..., min_length=1)


class QuestionResult(BaseModel):
    """Per-question grading outcome.

    Attributes:
        id: Question id.
        correct: Whether the learner's answer matched.
        user_answer: What the learner submitted (raw string or index).
        correct_answer: The expected answer.
        explanation: Pedagogical explanation surfaced from the question.
        source_refs: Chunk ids justifying the answer (grounded mode).
    """

    id: int
    correct: bool
    user_answer: Union[int, str, None]
    correct_answer: Union[int, str]
    explanation: str
    source_refs: Optional[List[str]] = None


class GradeReport(BaseModel):
    """Aggregate grading output.

    Attributes:
        score: Number of correct answers.
        total: Total number of questions.
        percent: ``score / total * 100`` rounded to one decimal.
        per_question: Detailed per-question outcomes in order.
    """

    score: int
    total: int
    percent: float
    per_question: List[QuestionResult]


def _strip_code_fences(text: str) -> str:
    """Strip ```json ...``` fences if the model wrapped its output anyway.

    Args:
        text: Raw model output.

    Returns:
        Inner JSON string, with surrounding fences removed.
    """
    fenced = re.match(
        r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL
    )
    return fenced.group(1) if fenced else text


def _parse_quiz_payload(payload: Any) -> Quiz:
    """Coerce raw model output into a :class:`Quiz`.

    Accepts either an already-validated ``Quiz`` (from
    ``with_structured_output``), a ``dict``, or a JSON string.

    Args:
        payload: Whatever the LLM returned.

    Returns:
        A validated :class:`Quiz`.

    Raises:
        ValueError: If the payload can't be parsed or validated.
    """
    if isinstance(payload, Quiz):
        return payload
    if isinstance(payload, dict):
        return Quiz.model_validate(payload)
    if isinstance(payload, str):
        cleaned = _strip_code_fences(payload)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Quiz output is not valid JSON: {e}") from e
        return Quiz.model_validate(data)
    # LangChain message? Pull .content.
    content = getattr(payload, "content", None)
    if content is not None:
        return _parse_quiz_payload(content)
    raise ValueError(f"Unrecognized quiz payload type: {type(payload)!r}")


def generate_quiz(
    llm: Any,
    topic: str,
    num_questions: int = 5,
    difficulty: Difficulty = "medium",
    question_types: QuestionTypesArg = "mixed",
    context_chunks: Optional[Sequence[RetrievedChunk]] = None,
    max_retries: int = 2,
) -> Quiz:
    """Generate a structured quiz using an LLM.

    Tries ``with_structured_output`` first for native schema enforcement,
    then falls back to a parse-and-validate loop with up to ``max_retries``
    attempts.

    Args:
        llm: A LangChain chat model.
        topic: Subject of the quiz.
        num_questions: Question count, clamped to ``[1, 50]``.
        difficulty: ``easy`` / ``medium`` / ``hard``.
        question_types: Type mix; ``mixed`` distributes evenly.
        context_chunks: Retrieved chunks for grounded mode. ``None`` /
            empty triggers ungrounded mode.
        max_retries: Extra fallback attempts if structured output fails.

    Returns:
        A validated :class:`Quiz`.

    Raises:
        ValueError: If the LLM repeatedly fails to produce valid output.

    Example:
        >>> quiz = generate_quiz(llm, "Photosynthesis", num_questions=5)
        >>> assert len(quiz.questions) == 5
    """
    num_questions = max(1, min(50, int(num_questions)))
    context_block = format_context_block(context_chunks or [])
    user_msg = build_quiz_user_message(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        question_types=question_types,
        context_block=context_block,
    )
    messages = [("system", QUIZ_SYSTEM_PROMPT), ("user", user_msg)]

    # First attempt — structured output (works on Anthropic/OpenAI/Google/Groq).
    try:
        structured = llm.with_structured_output(Quiz)
        result = structured.invoke(messages)
        return _parse_quiz_payload(result)
    except Exception as e:
        logger.warning("structured-output path failed (%s); falling back", e)

    # Fallback — plain invoke + manual parsing with retries.
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            response = llm.invoke(messages)
            return _parse_quiz_payload(response)
        except Exception as e:
            last_err = e
            logger.warning("Quiz parse attempt %d failed: %s", attempt, e)
            messages = [
                ("system", QUIZ_SYSTEM_PROMPT),
                ("user", user_msg),
                (
                    "user",
                    "Your previous output was not valid JSON matching the "
                    "schema. Return ONLY the JSON object, no prose, no "
                    "fences.",
                ),
            ]
    raise ValueError(
        f"Failed to generate a valid quiz after {max_retries + 1} attempts: "
        f"{last_err}"
    )


def _normalize_short_answer(s: str) -> str:
    """Normalize a short-answer string for comparison.

    Lowercases, collapses whitespace, strips punctuation at the edges, and
    removes leading articles ("a", "an", "the").

    Args:
        s: Raw user input.

    Returns:
        Canonicalized form.
    """
    s = s.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):]
    return s


def _grade_one(
    q: Question, user_answer: Union[int, str, None]
) -> QuestionResult:
    """Grade a single question.

    Args:
        q: The question definition.
        user_answer: The learner's submitted answer.

    Returns:
        A :class:`QuestionResult` describing the outcome.
    """
    correct = False
    if user_answer is None or user_answer == "":
        correct = False
    elif q.type in {"multiple_choice", "true_false"}:
        try:
            correct = int(user_answer) == int(q.correct_answer)
        except (TypeError, ValueError):
            correct = False
    else:  # short_answer
        expected = _normalize_short_answer(str(q.correct_answer))
        got = _normalize_short_answer(str(user_answer))
        if not expected:
            correct = False
        elif got == expected:
            correct = True
        else:
            # Whole-word/phrase match — expected must appear as a contiguous
            # sequence of words inside got. Prevents single-letter expected
            # answers from matching any text that happens to contain them.
            correct = bool(
                re.search(rf"\b{re.escape(expected)}\b", got)
            )

    return QuestionResult(
        id=q.id,
        correct=correct,
        user_answer=user_answer,
        correct_answer=q.correct_answer,
        explanation=q.explanation,
        source_refs=q.source_refs,
    )


def grade_quiz(
    quiz: Quiz, answers: dict[int, Union[int, str, None]]
) -> GradeReport:
    """Grade a quiz given a mapping of ``question_id -> answer``.

    Args:
        quiz: The quiz to grade.
        answers: Mapping from each question's ``id`` to the learner's
            submitted answer.

    Returns:
        A :class:`GradeReport` with score, percent, and per-question detail.
    """
    per_question = [_grade_one(q, answers.get(q.id)) for q in quiz.questions]
    score = sum(1 for r in per_question if r.correct)
    total = len(per_question)
    percent = round((score / total) * 100, 1) if total else 0.0
    return GradeReport(
        score=score, total=total, percent=percent, per_question=per_question
    )
