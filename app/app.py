"""Smart Teacher — Streamlit entry point.

This module is the only UI surface. It composes the provider factories, the
RAG pipeline, and the quiz module into a single application:

* **Sidebar** — provider/model/key selection, retrieval tuning, document
  upload.
* **Chat tab** — pedagogical Q&A grounded in the uploaded documents (or
  ungrounded if none).
* **Quiz tab** — configure, generate, take, and grade a quiz.

Run locally::

    streamlit run app/app.py
"""

from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, List, Optional

# Silence harmless transformers 5.x deprecation chatter from sentence-transformers'
# eager imports (zoedepth __path__ alias). We never touch those modules.
warnings.filterwarnings(
    "ignore", message=r".*Accessing `__path__`.*", module=r"transformers.*"
)

import streamlit as st

# Make sibling modules importable both as ``streamlit run app/app.py`` and
# ``streamlit run app.py`` inside the ``app/`` directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from embeddings_factory import get_embeddings, list_embedding_backends  # noqa: E402
from llm_factory import PROVIDERS, get_llm, list_providers  # noqa: E402
from prompts import build_teacher_messages  # noqa: E402
from quiz import GradeReport, Quiz, generate_quiz, grade_quiz  # noqa: E402
from rag import (  # noqa: E402
    RetrievedChunk,
    build_vector_store,
    fingerprint,
    format_context_block,
    load_uploaded_files,
    retrieve,
    split_documents,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_teacher")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model…")
def _cached_embeddings(backend: str, model: str, key_hint: str) -> Any:
    """Cache the embedding model. ``key_hint`` participates in the cache key.

    Args:
        backend: Embeddings backend id.
        model: Model identifier.
        key_hint: Truncated hint of the API key so re-keying invalidates the
            cache without storing the raw secret.

    Returns:
        A LangChain ``Embeddings`` instance.
    """
    return get_embeddings(backend=backend, model=model)


@st.cache_resource(show_spinner="Indexing documents…")
def _cached_vector_store(
    index_fingerprint: str,
    chunk_size: int,
    chunk_overlap: int,
    backend: str,
    model: str,
    _files: List[Any],
    _embeddings: Any,
) -> Any:
    """Build (and cache) a FAISS index for the given files.

    Args:
        index_fingerprint: Stable hash derived from files + chunk params; the
            primary cache key.
        chunk_size: Splitter chunk size.
        chunk_overlap: Splitter chunk overlap.
        backend: Embeddings backend id (cache participation only).
        model: Embeddings model identifier (cache participation only).
        _files: Streamlit ``UploadedFile`` list. Leading underscore tells
            Streamlit not to hash this argument.
        _embeddings: The embeddings instance. Same hashing exclusion.

    Returns:
        A FAISS vector store.
    """
    docs = load_uploaded_files(_files)
    if not docs:
        raise ValueError(
            "No readable content found in the uploaded files."
        )
    chunks = split_documents(
        docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return build_vector_store(chunks, _embeddings)


@st.cache_resource(show_spinner="Connecting to LLM…")
def _cached_llm(
    provider: str,
    model: str,
    temperature: float,
    key_hint: str,
) -> Any:
    """Cache the LLM client.

    Args:
        provider: Provider id.
        model: Model identifier.
        temperature: Sampling temperature.
        key_hint: Truncated hint of the key so re-keying invalidates the
            cache without storing the secret.

    Returns:
        A configured LangChain chat model.
    """
    return get_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=st.session_state.get(f"key::{provider}"),
    )


def _key_hint(value: Optional[str]) -> str:
    """Derive a non-sensitive cache hint from an API key.

    Args:
        value: The raw key, or ``None``.

    Returns:
        First/last 2 chars joined by ``…`` (or ``"none"``).
    """
    if not value:
        return "none"
    return f"{value[:2]}…{value[-2:]}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> dict[str, Any]:
    """Render the sidebar and return the active configuration dict.

    Returns:
        A dict containing ``provider``, ``model``, ``temperature``, ``k``,
        ``chunk_size``, ``chunk_overlap``, ``embed_backend``,
        ``embed_model``, ``uploaded_files``.
    """
    st.sidebar.title("⚙️ Settings")

    st.sidebar.subheader("LLM provider")
    providers = list_providers()
    provider_keys = list(providers.keys())
    provider_labels = [providers[k].label for k in provider_keys]
    default_idx = (
        provider_keys.index(st.session_state.get("provider", "groq"))
        if st.session_state.get("provider", "groq") in provider_keys
        else 0
    )
    provider_label = st.sidebar.selectbox(
        "Provider",
        provider_labels,
        index=default_idx,
        help="Pick any supported LLM provider. Switching here re-creates the "
        "client.",
    )
    provider = provider_keys[provider_labels.index(provider_label)]
    st.session_state["provider"] = provider
    spec = providers[provider]

    model = st.sidebar.text_input(
        "Model",
        value=st.session_state.get(f"model::{provider}", spec.default_model),
        help=f"Default: {spec.default_model}. {spec.notes}",
    )
    st.session_state[f"model::{provider}"] = model

    if spec.needs_key:
        api_key = st.sidebar.text_input(
            f"{spec.label} API key",
            value=st.session_state.get(f"key::{provider}", ""),
            type="password",
            help=f"Reads ${spec.env_var} from env / secrets if blank.",
        )
        st.session_state[f"key::{provider}"] = api_key
    else:
        st.sidebar.info(f"No API key needed for {spec.label}.")

    temperature = st.sidebar.slider(
        "Temperature", 0.0, 1.5, value=0.2, step=0.1
    )

    st.sidebar.subheader("Retrieval")
    k = st.sidebar.slider("Top-k chunks", 1, 12, value=4)
    chunk_size = st.sidebar.slider("Chunk size", 200, 2000, value=1000, step=50)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, value=150, step=10)

    with st.sidebar.expander("Embedding backend", expanded=False):
        backends = list_embedding_backends()
        bkeys = list(backends.keys())
        blabels = [backends[k].label for k in bkeys]
        sel = st.selectbox("Backend", blabels, index=0)
        embed_backend = bkeys[blabels.index(sel)]
        embed_model = st.text_input(
            "Embedding model",
            value=backends[embed_backend].default_model,
        )

    st.sidebar.subheader("Knowledge base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF / TXT / MD files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "markdown"],
        help="Without uploads the app runs in *ungrounded* topic mode.",
    )
    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} file(s) ready to index.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state["chat_history"] = []
        st.toast("Chat cleared.")

    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "k": k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_backend": embed_backend,
        "embed_model": embed_model,
        "uploaded_files": uploaded_files or [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_llm_safe(cfg: dict[str, Any]) -> Optional[Any]:
    """Get the LLM client, surfacing friendly errors in the UI.

    Args:
        cfg: Sidebar configuration dict.

    Returns:
        The LLM instance, or ``None`` if construction failed.
    """
    try:
        return _cached_llm(
            provider=cfg["provider"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            key_hint=_key_hint(
                st.session_state.get(f"key::{cfg['provider']}")
            ),
        )
    except ValueError as e:
        st.error(str(e))
    except ImportError as e:
        st.error(
            f"Provider package missing: {e}. Install it via "
            "`pip install -r requirements.txt`."
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        logger.exception("LLM init failed")
    return None


def _get_vector_store_safe(cfg: dict[str, Any]) -> Optional[Any]:
    """Build the vector store from current uploads, with UI error handling.

    Args:
        cfg: Sidebar configuration dict.

    Returns:
        The vector store, or ``None`` if no files / on failure.
    """
    files = cfg["uploaded_files"]
    if not files:
        return None
    try:
        embeddings = _cached_embeddings(
            backend=cfg["embed_backend"],
            model=cfg["embed_model"],
            key_hint=_key_hint(
                st.session_state.get(f"key::{cfg['provider']}")
            ),
        )
        fp = fingerprint(files, cfg["chunk_size"], cfg["chunk_overlap"])
        return _cached_vector_store(
            index_fingerprint=fp,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            backend=cfg["embed_backend"],
            model=cfg["embed_model"],
            _files=files,
            _embeddings=embeddings,
        )
    except Exception as e:
        st.error(f"Failed to index documents: {e}")
        logger.exception("Indexing failed")
        return None


def _render_sources(chunks: List[RetrievedChunk]) -> None:
    """Render an expandable source-citation block.

    Args:
        chunks: Retrieved chunks to display.
    """
    if not chunks:
        return
    with st.expander(f"📚 Sources used ({len(chunks)})", expanded=False):
        for c in chunks:
            score_txt = (
                f" · similarity {c.score:.3f}" if c.score is not None else ""
            )
            st.markdown(
                f"**`{c.chunk_id}`**  — *{c.source}*{score_txt}"
            )
            preview = c.content.strip().replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "…"
            st.caption(preview)
            st.markdown("---")


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------


def _render_chat_tab(cfg: dict[str, Any]) -> None:
    """Render the chat UI.

    Args:
        cfg: Sidebar configuration dict.
    """
    st.subheader("💬 Ask your Smart Teacher")
    st.caption(
        "Ask anything about your topic. With documents uploaded, answers "
        "are grounded and cited. Without, answers are clearly marked "
        "ungrounded."
    )

    history: list[dict[str, Any]] = st.session_state.setdefault(
        "chat_history", []
    )
    vector_store = _get_vector_store_safe(cfg)

    with st.expander("💡 Example prompts", expanded=False):
        st.markdown(
            "- *Teach me linear algebra from scratch in 2 weeks.*\n"
            "- *Summarize the main argument of the uploaded paper.*\n"
            "- *I want to learn Spanish verb conjugations — what's the "
            "best method?*\n"
            "- *Compare gradient descent and Adam from the lecture notes.*"
        )

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    user_input = st.chat_input("Type a question or topic…")
    if not user_input:
        return

    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    llm = _get_llm_safe(cfg)
    if llm is None:
        return

    sources: List[RetrievedChunk] = []
    if vector_store is not None:
        try:
            sources = retrieve(vector_store, user_input, k=cfg["k"])
        except Exception as e:
            st.warning(f"Retrieval failed: {e}")
            logger.exception("Retrieval failed")

    context_block = format_context_block(sources)
    prior = [
        (m["role"], m["content"])
        for m in history[:-1]
        if m["role"] in {"user", "assistant"}
    ]
    messages = build_teacher_messages(
        question=user_input,
        context_block=context_block,
        history=prior[-6:],  # keep last 3 exchanges to bound tokens
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Thinking…"):
                response = llm.invoke(messages)
            answer = getattr(response, "content", str(response))
        except Exception as e:
            err = _friendly_llm_error(e)
            placeholder.error(err)
            history.append(
                {"role": "assistant", "content": err, "sources": []}
            )
            return

        placeholder.markdown(answer)
        _render_sources(sources)

    history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def _friendly_llm_error(exc: Exception) -> str:
    """Translate raw LLM exceptions into actionable user-facing text.

    Args:
        exc: The caught exception.

    Returns:
        A short Markdown error message.
    """
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg:
        return (
            "⚠️ Provider rate limit hit. Wait a moment, lower the "
            "temperature, or switch provider in the sidebar."
        )
    if "auth" in msg or "401" in msg or "invalid api key" in msg:
        return (
            "⚠️ Authentication failed. Re-check the API key in the sidebar."
        )
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return (
            "⚠️ Network error reaching the provider. Check your connection "
            "or try Ollama for local inference."
        )
    return (
        f"⚠️ Unexpected error: `{exc}`. See logs for the traceback.\n\n"
        f"```\n{traceback.format_exc(limit=2)}\n```"
    )


# ---------------------------------------------------------------------------
# Quiz tab
# ---------------------------------------------------------------------------


def _render_quiz_tab(cfg: dict[str, Any]) -> None:
    """Render the quiz UI.

    Args:
        cfg: Sidebar configuration dict.
    """
    st.subheader("📝 Generate a quiz")
    st.caption(
        "Quiz yourself on the current topic. Grounded mode pulls "
        "questions from your uploaded documents."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_q = st.number_input(
            "Questions", min_value=3, max_value=20, value=5, step=1
        )
    with col2:
        difficulty = st.selectbox(
            "Difficulty", ["easy", "medium", "hard"], index=1
        )
    with col3:
        qtype = st.selectbox(
            "Question types",
            ["mixed", "multiple_choice", "true_false", "short_answer"],
            index=0,
        )
    with col4:
        grounded = st.checkbox(
            "Ground in docs",
            value=bool(cfg["uploaded_files"]),
            disabled=not cfg["uploaded_files"],
            help="Requires uploaded documents.",
        )

    topic = st.text_input(
        "Quiz topic",
        value=st.session_state.get("quiz_topic", ""),
        placeholder="e.g. 'photosynthesis' or 'chapter 3 of the uploaded "
        "paper'",
    )
    st.session_state["quiz_topic"] = topic

    if st.button("🎯 Generate quiz", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            _generate_and_store_quiz(
                cfg=cfg,
                topic=topic.strip(),
                num_questions=int(num_q),
                difficulty=difficulty,
                question_types=qtype,
                grounded=grounded,
            )

    if st.session_state.get("quiz") is not None:
        _render_active_quiz()


def _generate_and_store_quiz(
    cfg: dict[str, Any],
    topic: str,
    num_questions: int,
    difficulty: str,
    question_types: str,
    grounded: bool,
) -> None:
    """Generate a quiz and stash it into session state.

    Args:
        cfg: Sidebar configuration dict.
        topic: Quiz subject.
        num_questions: How many questions.
        difficulty: easy/medium/hard.
        question_types: type or "mixed".
        grounded: Whether to ground in uploaded docs.
    """
    llm = _get_llm_safe(cfg)
    if llm is None:
        return
    chunks: list[RetrievedChunk] = []
    if grounded:
        vs = _get_vector_store_safe(cfg)
        if vs is None:
            st.warning("No indexed documents — falling back to ungrounded.")
        else:
            try:
                # Pull a wider net for quiz coverage.
                chunks = retrieve(vs, topic, k=max(cfg["k"], 8))
            except Exception as e:
                st.warning(f"Retrieval failed: {e} — running ungrounded.")
    try:
        with st.spinner("Generating quiz…"):
            quiz = generate_quiz(
                llm=llm,
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,  # type: ignore[arg-type]
                question_types=question_types,  # type: ignore[arg-type]
                context_chunks=chunks,
            )
    except Exception as e:
        st.error(_friendly_llm_error(e))
        return

    st.session_state["quiz"] = quiz
    st.session_state["quiz_answers"] = {}
    st.session_state["quiz_report"] = None
    st.session_state["quiz_sources"] = chunks
    st.toast(f"Generated {len(quiz.questions)} question(s).")


def _render_active_quiz() -> None:
    """Render the active quiz form, grading, and retry actions."""
    quiz: Quiz = st.session_state["quiz"]
    answers: dict[int, Any] = st.session_state.setdefault("quiz_answers", {})
    report: Optional[GradeReport] = st.session_state.get("quiz_report")
    retry_only: set[int] = set(st.session_state.get("retry_ids", set()))

    st.markdown("---")
    st.markdown(f"### {quiz.topic}  · _{quiz.difficulty}_")

    with st.form("quiz_form", clear_on_submit=False):
        for q in quiz.questions:
            if retry_only and q.id not in retry_only:
                continue
            st.markdown(f"**Q{q.id}.** {q.prompt}")
            key = f"answer_{q.id}"
            if q.type == "multiple_choice" and q.options:
                choice = st.radio(
                    f"Choose for Q{q.id}",
                    options=list(range(len(q.options))),
                    format_func=lambda i, opts=q.options: f"{chr(65 + i)}. {opts[i]}",
                    key=key,
                    label_visibility="collapsed",
                    index=None,
                )
                answers[q.id] = choice
            elif q.type == "true_false":
                choice = st.radio(
                    f"Choose for Q{q.id}",
                    options=[0, 1],
                    format_func=lambda i: "True" if i == 0 else "False",
                    key=key,
                    label_visibility="collapsed",
                    index=None,
                    horizontal=True,
                )
                answers[q.id] = choice
            else:  # short_answer
                answers[q.id] = st.text_input(
                    f"Your answer for Q{q.id}",
                    key=key,
                    label_visibility="collapsed",
                )
            st.markdown("")  # spacer
        submitted = st.form_submit_button("✅ Submit answers", type="primary")

    if submitted:
        st.session_state["quiz_answers"] = answers
        report = grade_quiz(quiz, answers)
        st.session_state["quiz_report"] = report

    if report is not None:
        _render_grade_report(quiz, report)


def _render_grade_report(quiz: Quiz, report: GradeReport) -> None:
    """Show the score banner, per-question feedback, and retry controls.

    Args:
        quiz: The quiz that was graded.
        report: The grading outcome.
    """
    st.markdown("---")
    pct = report.percent
    if pct >= 80:
        st.success(
            f"🎉 Score: **{report.score}/{report.total}** ({pct:.1f}%)"
        )
    elif pct >= 50:
        st.info(f"Score: **{report.score}/{report.total}** ({pct:.1f}%)")
    else:
        st.warning(f"Score: **{report.score}/{report.total}** ({pct:.1f}%)")

    for r in report.per_question:
        q = next(q for q in quiz.questions if q.id == r.id)
        icon = "✅" if r.correct else "❌"
        with st.expander(f"{icon} Q{r.id}. {q.prompt}", expanded=not r.correct):
            if q.type in {"multiple_choice", "true_false"} and q.options:
                try:
                    correct_label = q.options[int(q.correct_answer)]
                except (TypeError, ValueError, IndexError):
                    correct_label = str(q.correct_answer)
                st.markdown(f"**Correct answer:** {correct_label}")
                if r.user_answer is not None:
                    try:
                        user_label = q.options[int(r.user_answer)]
                    except (TypeError, ValueError, IndexError):
                        user_label = str(r.user_answer)
                    st.markdown(f"**Your answer:** {user_label}")
                else:
                    st.markdown("**Your answer:** _(blank)_")
            else:
                st.markdown(f"**Correct answer:** {q.correct_answer}")
                st.markdown(
                    f"**Your answer:** {r.user_answer or '_(blank)_'}"
                )
            st.markdown(f"**Why:** {q.explanation}")
            if r.source_refs:
                st.markdown(
                    "**Sources:** "
                    + ", ".join(f"`{s}`" for s in r.source_refs)
                )

    if st.session_state.get("quiz_sources"):
        _render_sources(st.session_state["quiz_sources"])

    col_a, col_b = st.columns(2)
    with col_a:
        wrong_ids = [r.id for r in report.per_question if not r.correct]
        if st.button(
            "🔁 Retry incorrect only",
            disabled=not wrong_ids,
            use_container_width=True,
        ):
            st.session_state["retry_ids"] = set(wrong_ids)
            st.session_state["quiz_report"] = None
            st.rerun()
    with col_b:
        if st.button("🆕 New quiz", use_container_width=True):
            for k in (
                "quiz",
                "quiz_answers",
                "quiz_report",
                "quiz_sources",
                "retry_ids",
            ):
                st.session_state.pop(k, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit entry point: render header, sidebar, and tabs."""
    st.set_page_config(
        page_title="Smart Teacher",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🎓 Smart Teacher")
    st.caption(
        "An LLM-agnostic AI tutor with retrieval-grounded answers and "
        "on-demand quizzes."
    )

    cfg = _render_sidebar()

    tab_chat, tab_quiz = st.tabs(["💬 Chat", "📝 Quiz"])
    with tab_chat:
        _render_chat_tab(cfg)
    with tab_quiz:
        _render_quiz_tab(cfg)


if __name__ == "__main__":
    main()
