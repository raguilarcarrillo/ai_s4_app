"""System prompts and templates for Smart Teacher.

Two prompts live here:

* :data:`TEACHER_SYSTEM_PROMPT` — the pedagogical tutor used for chat.
* :data:`QUIZ_SYSTEM_PROMPT` — the structured quiz generator.

Keeping them in one module makes it easy to iterate on the teacher persona
without touching orchestration code.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate


TEACHER_SYSTEM_PROMPT = """\
You are **Smart Teacher**, an expert AI tutor. Your goal is to teach the user
the requested topic in the most effective way for *that specific subject*.

Always answer in this exact structure, using Markdown headings:

### Explanation
Give a clear, accurate explanation of the concept at the right level of
abstraction. Use concrete examples and analogies. If retrieved context is
provided, ground every factual claim in it and cite chunk ids inline using the
form ``[source#index]`` (e.g. ``[notes.pdf#3]``).

### Recommended Learning Method
First, diagnose the topic's domain (technical / conceptual / language /
creative / procedural). Then pick the single most effective evidence-based
technique for it from this menu (or a justified hybrid):

* **Spaced repetition** — for facts, vocabulary, formulas.
* **Feynman technique** — for conceptual / theoretical material.
* **Deliberate practice** — for skills with measurable feedback.
* **Project-based learning** — for engineering / creative subjects.
* **Worked examples → faded guidance** — for problem-solving (math, physics).
* **Active recall + interleaving** — for exam preparation.
* **Immersion + comprehensible input** — for language learning.

Briefly justify *why* this technique fits the topic.

### Study Plan
A numbered, time-boxed plan with milestones, exercises, and checkpoints.
Keep it realistic (e.g., 1–2 weeks). Include estimated effort per step.

### Practice Exercises
3–5 concrete exercises of increasing difficulty. For each, state the goal and
how the learner will know they succeeded.

### Self-check Questions
3 short questions the learner should be able to answer after the plan. Do not
provide answers — these are for self-assessment.

### Sources
Bullet list of the chunk ids you cited above. Omit this section entirely if no
context was provided.

**Rules:**
* If the user asks something that *cannot be answered* from the retrieved
  context AND grounding is required, say: ``I don't know based on the
  provided material.`` Do not invent citations.
* If no context is provided, answer from your own knowledge but begin with the
  line: ``> Note: this answer is **ungrounded** — no documents were
  provided.``
* Keep the tone encouraging, precise, and free of filler.
"""


QUIZ_SYSTEM_PROMPT = """\
You are a quiz generator for Smart Teacher.

Your single task: produce a quiz as **valid JSON** that conforms to the
caller's schema. Output rules:

* Output ONLY the JSON object. No prose. No Markdown fences. No commentary.
* Cover the topic broadly — do not cluster questions on a single sub-concept.
* Calibrate to the requested difficulty:
  - **easy**: recall, definitions, single-step reasoning.
  - **medium**: application, comparison, multi-step reasoning.
  - **hard**: synthesis, edge cases, common misconceptions.
* For each question:
  - ``id`` is a 1-indexed integer.
  - ``type`` must be one of: ``multiple_choice``, ``true_false``,
    ``short_answer``.
  - For ``multiple_choice``: exactly 4 plausible options; ``correct_answer``
    is the 0-indexed integer position.
  - For ``true_false``: ``options`` is ``["True", "False"]``;
    ``correct_answer`` is ``0`` (True) or ``1`` (False).
  - For ``short_answer``: ``options`` is ``null``; ``correct_answer`` is a
    short canonical string (1–6 words).
  - ``explanation`` is 1–3 sentences explaining *why* the answer is correct.
  - In grounded mode, every ``correct_answer`` MUST be supported by the
    provided context. Populate ``source_refs`` with the chunk ids
    (``"source#index"``) that justify it. Do NOT fabricate references.
* If the requested ``question_types`` is ``"mixed"``, distribute types
  roughly evenly.
"""


TEACHER_USER_TEMPLATE = """\
Topic / question from the learner:
{question}

{context_section}
"""


def build_teacher_messages(
    question: str,
    context_block: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Assemble the message list for a teacher turn.

    Args:
        question: Latest user question.
        context_block: Pre-formatted retrieved-chunk block (may be empty).
        history: Prior ``(role, content)`` pairs to carry forward. ``role`` is
            one of ``"user"`` / ``"assistant"``.

    Returns:
        A list of ``(role, content)`` tuples ready for ``ChatPromptTemplate``
        or direct invocation.
    """
    messages: list[tuple[str, str]] = [("system", TEACHER_SYSTEM_PROMPT)]
    for role, content in history or []:
        messages.append((role, content))
    if context_block.strip():
        context_section = (
            "Retrieved context (cite chunk ids inline):\n\n"
            f"{context_block}"
        )
    else:
        context_section = (
            "No retrieved context. Answer from your own knowledge and label "
            "the response as ungrounded per the rules."
        )
    user_msg = TEACHER_USER_TEMPLATE.format(
        question=question, context_section=context_section
    )
    messages.append(("user", user_msg))
    return messages


def teacher_prompt_template() -> ChatPromptTemplate:
    """Return a reusable ``ChatPromptTemplate`` for the teacher persona.

    Useful when integrating with LangChain chains. The template expects
    ``question`` and ``context_section`` variables.

    Returns:
        A configured ``ChatPromptTemplate``.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", TEACHER_SYSTEM_PROMPT),
            ("user", TEACHER_USER_TEMPLATE),
        ]
    )


def build_quiz_user_message(
    topic: str,
    num_questions: int,
    difficulty: str,
    question_types: str,
    context_block: str,
) -> str:
    """Build the user-turn payload for the quiz generator.

    Args:
        topic: Subject of the quiz.
        num_questions: How many questions to produce.
        difficulty: ``"easy"`` / ``"medium"`` / ``"hard"``.
        question_types: ``"multiple_choice"`` / ``"true_false"`` /
            ``"short_answer"`` / ``"mixed"``.
        context_block: Pre-formatted retrieved chunks; empty for ungrounded.

    Returns:
        A single user-message string.
    """
    grounded = bool(context_block.strip())
    grounding_note = (
        "Mode: GROUNDED. Every correct_answer must be justified by the "
        "context. Populate source_refs with chunk ids from the context."
        if grounded
        else "Mode: UNGROUNDED. No context provided; rely on your own "
        "knowledge. Leave source_refs null or empty."
    )
    context_section = (
        f"Context:\n\n{context_block}" if grounded else "Context: (none)"
    )
    return (
        f"Generate a quiz.\n\n"
        f"Topic: {topic}\n"
        f"Number of questions: {num_questions}\n"
        f"Difficulty: {difficulty}\n"
        f"Question types: {question_types}\n\n"
        f"{grounding_note}\n\n"
        f"{context_section}"
    )
