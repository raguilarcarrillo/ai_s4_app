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

import hashlib
import logging
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional

# Load .env from the project root (one level up from app/) before any
# downstream module reads os.environ. Silent no-op if .env is absent or
# python-dotenv isn't installed.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Silence harmless transformers 5.x deprecation chatter from sentence-transformers'
# eager imports (zoedepth __path__ alias). We never touch those modules.
# Two layers: warnings.warn (used by some submodules) + logging (used by the
# lazy-module __getattr__ that emits the __path__ alias notice).
warnings.filterwarnings(
    "ignore", message=r".*Accessing `__path__`.*", module=r"transformers.*"
)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

# Make sibling modules importable both as ``streamlit run app/app.py`` and
# ``streamlit run app.py`` inside the ``app/`` directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import charts  # noqa: E402
import persistence  # noqa: E402
from embeddings_factory import get_embeddings, list_embedding_backends  # noqa: E402
from llm_factory import PROVIDERS, get_llm, list_providers  # noqa: E402
from pdf_export import markdown_to_pdf_bytes, pdf_filename  # noqa: E402
from prompts import build_teacher_messages  # noqa: E402
from quiz import GradeReport, Quiz, generate_quiz, grade_quiz  # noqa: E402
from rag import (  # noqa: E402
    RetrievedChunk,
    build_vector_store,
    fingerprint,
    format_context_block,
    load_disk_files,
    load_uploaded_files,
    retrieve,
    split_documents,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_teacher")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


# Embeddings have no user-supplied secrets in the current design — they
# either run locally (sentence-transformers) or read env vars set at deploy
# time. A process-wide cache is therefore safe and avoids re-loading the
# ~80 MB local model per browser session.
@st.cache_resource(show_spinner="Loading embedding model…")
def _cached_embeddings(backend: str, model: str) -> Any:
    """Cache the embedding model.

    Args:
        backend: Embeddings backend id.
        model: Model identifier.

    Returns:
        A LangChain ``Embeddings`` instance.
    """
    return get_embeddings(backend=backend, model=model)


# Indexing is keyed by a sha256 fingerprint of file bytes + chunk params, so
# cache hits across sessions only occur on byte-identical inputs. No user
# secret participates.
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
        _files: Either Streamlit ``UploadedFile`` objects (fresh upload) or
            :class:`pathlib.Path` objects (restored from disk). Leading
            underscore tells Streamlit not to hash this argument.
        _embeddings: The embeddings instance. Same hashing exclusion.

    Returns:
        A FAISS vector store.
    """
    if not _files:
        raise ValueError("No files provided.")
    if isinstance(_files[0], Path):
        docs = load_disk_files(_files)
    else:
        docs = load_uploaded_files(_files)
    if not docs:
        raise ValueError(
            "No readable content found in the uploaded files."
        )
    chunks = split_documents(
        docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return build_vector_store(chunks, _embeddings)


# Cleanup runs once per process via @st.cache_resource memoization. Wrapped
# in try/except so a misconfigured cache dir never blocks app startup.
@st.cache_resource(show_spinner=False)
def _run_startup_cleanup() -> bool:
    try:
        removed = persistence.cleanup_expired()
        if removed:
            logger.info(
                "Persistence: cleaned %d expired session(s)", removed
            )
    except Exception:
        logger.exception("Persistence cleanup failed")
    return True


def _key_hint(value: Optional[str]) -> str:
    """Derive a non-sensitive cache hint from an API key.

    Uses sha256 so the hint is collision-resistant *and* unrecoverable —
    safe to put in cache keys, logs, or telemetry.

    Args:
        value: The raw key, or ``None``.

    Returns:
        16-char hex prefix of sha256(value), or ``"none"``.
    """
    if not value:
        return "none"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _get_llm_for_session(
    provider: str,
    model: str,
    temperature: float,
    api_key: Optional[str],
) -> Any:
    """Build (and per-session-cache) a chat LLM client.

    The cache lives in ``st.session_state`` so each browser session has its
    own client bound to its own key. This prevents the cross-tenant leak
    you'd get from ``@st.cache_resource`` (process-wide). An LRU bound of
    four entries keeps memory in check when the user flips providers /
    models repeatedly.

    Args:
        provider: Provider id.
        model: Model identifier.
        temperature: Sampling temperature.
        api_key: The session's API key for ``provider``.

    Returns:
        A configured LangChain chat model.
    """
    cache: "OrderedDict[str, Any]" = st.session_state.setdefault(
        "_llm_cache", OrderedDict()
    )
    cache_key = (
        f"{provider}::{model}::{temperature:.4f}::{_key_hint(api_key)}"
    )
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    with st.spinner("Connecting to LLM…"):
        client = get_llm(
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
    cache[cache_key] = client
    while len(cache) > 4:
        cache.popitem(last=False)
    return client


def _redact_secrets(text: str) -> str:
    """Replace any session API key occurrences in ``text`` with a placeholder.

    Belt-and-suspenders: provider SDK errors occasionally embed the bearer
    token in the message. We strip those before rendering anything to the
    UI so the page never echoes a secret back at the user.
    """
    if not text:
        return text
    redacted = text
    for k, v in list(st.session_state.items()):
        if (
            isinstance(k, str)
            and k.startswith("key::")
            and isinstance(v, str)
            and v
        ):
            redacted = redacted.replace(v, "***REDACTED***")
    return redacted


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
        "Upload PDF / TXT / MD / IPYNB files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "markdown", "ipynb"],
        help="Without uploads the app runs in *ungrounded* topic mode. "
        "For notebooks, only markdown + code cells are indexed; outputs "
        "are skipped.",
    )
    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} file(s) ready to index.")

    _render_persistence_section()

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
        "persist_enabled": bool(st.session_state.get("persist_enabled", False)),
    }


def _render_persistence_section() -> None:
    """Render the opt-in persistence controls in the sidebar.

    Strict opt-in. Nothing is written to disk unless the user actively
    checks the "Persist this session" box. The on-disk session directory
    is ``sha256(token)`` so the raw token never lands on disk.
    """
    with st.sidebar.expander("💾 Persistence (opt-in)", expanded=False):
        st.caption(
            "Save your uploaded files + FAISS index to disk so you can "
            "restore this session later with a token. Off by default — "
            "nothing is persisted until you check the box."
        )

        st.checkbox(
            "Persist this session",
            value=st.session_state.get("persist_enabled", False),
            key="persist_enabled",
            help=(
                "When on, files + manifest are written to "
                "./.cache/sessions/. The directory name is the hash of "
                "your token, so listing the cache alone doesn't reveal it."
            ),
        )

        if st.session_state.get("persist_enabled"):
            if "persist_token" not in st.session_state:
                st.session_state["persist_token"] = persistence.new_token()
            st.markdown("**Your session token** — copy and keep it safe:")
            st.code(st.session_state["persist_token"], language="text")
            st.caption(
                "Anyone with this token can restore (and read) the "
                "session. Treat it like a share link."
            )

        st.markdown("---")
        restore_token = st.text_input(
            "Restore from token",
            value="",
            help="Paste a 32-char hex token from a previous session.",
            key="restore_input",
        )
        if st.button(
            "🔗 Load session",
            disabled=not restore_token.strip(),
            use_container_width=True,
        ):
            try:
                _, paths = persistence.load_session(restore_token.strip())
            except persistence.PersistenceError as e:
                st.error(str(e))
            else:
                st.session_state["restored_paths"] = paths
                st.session_state["persist_enabled"] = True
                st.session_state["persist_token"] = (
                    restore_token.strip().lower()
                )
                st.success(f"Restored {len(paths)} file(s).")
                st.rerun()

        restored = st.session_state.get("restored_paths") or []
        if restored:
            st.info(
                "Restored from previous session: "
                + ", ".join(p.name for p in restored)
            )
            if st.button(
                "🗑️ Clear restored session", use_container_width=True
            ):
                for k in ("restored_paths", "restore_input"):
                    st.session_state.pop(k, None)
                st.rerun()


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
        return _get_llm_for_session(
            provider=cfg["provider"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            api_key=st.session_state.get(f"key::{cfg['provider']}"),
        )
    except ValueError as e:
        st.error(_redact_secrets(str(e)))
    except ImportError as e:
        st.error(
            f"Provider package missing: {e}. Install it via "
            "`pip install -r requirements.txt`."
        )
    except Exception as e:
        st.error(_redact_secrets(f"Failed to initialize LLM: {e}"))
        logger.exception("LLM init failed")
    return None


def _get_vector_store_safe(cfg: dict[str, Any]) -> Optional[Any]:
    """Build the vector store from current uploads, with UI error handling.

    Sources are resolved with this precedence:
      1. Fresh uploads via the file_uploader widget take priority.
      2. Otherwise, files restored from a persisted session are used.
      3. If neither is present, returns ``None`` (ungrounded mode).

    Persistence layering (when ``cfg["persist_enabled"]`` is true):
      * The on-disk FAISS cache at ``./.cache/faiss/<fingerprint>/`` is
        checked first — a hit avoids any embedding work.
      * On a miss, the in-memory ``@st.cache_resource`` path runs, and the
        resulting FAISS index is then persisted to disk for next time.
      * For *fresh* uploads, raw file bytes + manifest are written under
        the per-session blob. Restored files are already on disk, so we
        only refresh their manifest's ``last_accessed_at``.

    Args:
        cfg: Sidebar configuration dict.

    Returns:
        The vector store, or ``None`` if no files / on failure.
    """
    uploaded = cfg.get("uploaded_files") or []
    restored: list[Path] = st.session_state.get("restored_paths") or []

    if uploaded:
        inputs: list[Any] = list(uploaded)
        is_restored = False
    elif restored:
        inputs = list(restored)
        is_restored = True
    else:
        return None

    persist_on = bool(cfg.get("persist_enabled"))

    try:
        embeddings = _cached_embeddings(
            backend=cfg["embed_backend"],
            model=cfg["embed_model"],
        )
        fp = fingerprint(inputs, cfg["chunk_size"], cfg["chunk_overlap"])

        vs: Optional[Any] = None
        # Disk cache first when persistence is on; also when we're already
        # restoring (the index is probably there from the original save).
        if persist_on or is_restored:
            try:
                vs = persistence.load_faiss(fp, embeddings)
            except Exception:
                logger.exception("FAISS disk-cache load failed")

        if vs is None:
            vs = _cached_vector_store(
                index_fingerprint=fp,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                backend=cfg["embed_backend"],
                model=cfg["embed_model"],
                _files=inputs,
                _embeddings=embeddings,
            )

        if persist_on:
            # Streamlit reruns _get_vector_store_safe on every interaction,
            # so we gate the actual disk writes behind a per-session marker.
            # Rewriting a 100 MB PDF on every chat message is otherwise a
            # real possibility.
            token = st.session_state.get("persist_token") or persistence.new_token()
            st.session_state["persist_token"] = token
            marker = f"persist_marker::{token}::{fp}"
            if marker not in st.session_state:
                _save_to_persistence(
                    vs, fp, inputs, is_restored=is_restored, cfg=cfg
                )
                st.session_state[marker] = True

        return vs
    except Exception as e:
        st.error(_redact_secrets(f"Failed to index documents: {e}"))
        logger.exception("Indexing failed")
        return None


def _save_to_persistence(
    vs: Any,
    fp: str,
    inputs: list[Any],
    *,
    is_restored: bool,
    cfg: dict[str, Any],
) -> None:
    """Persist the FAISS index and (for fresh uploads) the session blob.

    Errors are surfaced as a sidebar warning but never block indexing.
    The caller is responsible for ensuring this only runs once per
    ``(token, fingerprint)`` to avoid rewriting bytes on every Streamlit
    rerun.
    """
    token = st.session_state["persist_token"]  # caller guarantees this exists
    try:
        persistence.save_faiss(fp, vs)
        if not is_restored:
            payload = [
                (
                    Path(getattr(f, "name", "uploaded")).name,
                    bytes(f.getbuffer()),
                )
                for f in inputs
            ]
            persistence.save_session(
                token,
                payload,
                fingerprint=fp,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                embed_backend=cfg["embed_backend"],
                embed_model=cfg["embed_model"],
            )
    except persistence.PersistenceError as e:
        st.sidebar.warning(f"Persistence skipped: {e}")
    except Exception:
        logger.exception("Persistence save failed")
        st.sidebar.warning("Persistence save failed; see server logs.")


def _get_pdf_bytes(title: str, content: str) -> bytes:
    """Render content to PDF bytes, memoized in session state.

    The cache is per-browser-session and bounded so a long chat doesn't
    pin unbounded memory. Bytes-by-bytes generation is expensive enough
    that we'd otherwise pay for every Streamlit rerun.
    """
    cache: "OrderedDict[str, bytes]" = st.session_state.setdefault(
        "_pdf_cache", OrderedDict()
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    cache_key = f"{title}::{digest}"
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    pdf_bytes = markdown_to_pdf_bytes(title, content)
    cache[cache_key] = pdf_bytes
    while len(cache) > 32:
        cache.popitem(last=False)
    return pdf_bytes


def _render_pdf_save_button(
    title: str, content: str, *, key: str, label: str = "📄 Download as PDF"
) -> None:
    """Render a Streamlit download_button delivering the content as a PDF.

    The browser saves the file via its own download flow — no filesystem
    writes on the server, so this works on HuggingFace Spaces too.

    Args:
        title: PDF title (also used as the filename stem).
        content: Markdown body to render into the PDF.
        key: Streamlit widget key — must be unique per call site.
        label: Button text.
    """
    if not content or not content.strip():
        return
    try:
        pdf_bytes = _get_pdf_bytes(title, content)
    except Exception as e:
        st.warning(f"Could not generate PDF: {e}")
        logger.exception("PDF generation failed")
        return
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=pdf_filename(title, content),
        mime="application/pdf",
        key=f"pdfbtn_{key}",
    )


def _render_assistant_content(content: str, *, key_prefix: str) -> None:
    """Render an assistant reply, splitting out any inline chart blocks.

    Markdown surrounding the chart goes through ``st.markdown`` as before;
    chart blocks become interactive Plotly canvases. A chart block that
    fails to parse falls back to its raw fenced source plus a one-line
    caption so the user can see what the LLM tried to do.
    """
    if not content:
        return
    for i, (kind, payload) in enumerate(charts.split_text_and_charts(content)):
        if kind == "text":
            if payload.strip():
                st.markdown(payload)
        else:  # "chart"
            fig, err = charts.render(payload)
            if fig is not None:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"chart_{key_prefix}_{i}",
                )
            else:
                st.code(payload, language="json")
                st.caption(f"Chart could not be rendered: {err}")


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

    for idx, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_content(msg["content"], key_prefix=f"hist_{idx}")
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    _render_sources(msg["sources"])
                _render_pdf_save_button(
                    title=_pdf_title_for_chat(history, idx),
                    content=msg["content"],
                    key=f"chat_hist_{idx}",
                )

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
            logger.exception("LLM invoke failed")
            err = _redact_secrets(_friendly_llm_error(e))
            placeholder.error(err)
            history.append(
                {"role": "assistant", "content": err, "sources": []}
            )
            return

        # Clear the spinner placeholder before re-rendering so the assistant
        # block doesn't have a stale empty box above the actual content.
        placeholder.empty()
        _render_assistant_content(answer, key_prefix=f"new_{len(history)}")
        _render_sources(sources)
        _render_pdf_save_button(
            title=_truncate_for_title(user_input),
            content=answer,
            key=f"chat_new_{len(history)}",
        )

    history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def _truncate_for_title(text: str, limit: int = 80) -> str:
    """Trim a long prompt into a short, single-line PDF title."""
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _pdf_title_for_chat(history: list[dict[str, Any]], assistant_idx: int) -> str:
    """Pick the most recent user prompt before an assistant message as the title."""
    for i in range(assistant_idx - 1, -1, -1):
        if history[i].get("role") == "user":
            return _truncate_for_title(history[i].get("content", ""))
    return "Smart Teacher answer"


def _friendly_llm_error(exc: Exception) -> str:
    """Translate raw LLM exceptions into actionable user-facing text.

    Any returned string is passed through ``_redact_secrets`` so a provider
    SDK that embeds the bearer token in its error message can never echo
    that token to the page.

    Args:
        exc: The caught exception.

    Returns:
        A short Markdown error message.
    """
    # Non-ASCII header value almost always means the user pasted prompt
    # text into the API key field — HTTP forbids non-latin-1 in header
    # values, so the provider client raises UnicodeEncodeError before
    # the request leaves the machine.
    if isinstance(exc, UnicodeEncodeError):
        return (
            "⚠️ API key looks invalid — it contains non-ASCII characters. "
            "This usually means prompt text was pasted into the API key "
            "field by mistake. Clear the key in the sidebar and paste "
            "the actual key (e.g. `gsk_…` for Groq, `sk-…` for OpenAI, "
            "`sk-ant-…` for Anthropic, `AIza…` for Gemini, `hf_…` for "
            "HuggingFace)."
        )

    msg = str(exc).lower()
    # Some clients wrap the underlying UnicodeEncodeError; sniff the text.
    if "codec can't encode" in msg and "ascii" in msg:
        return (
            "⚠️ API key contains invalid characters. Re-check the key in "
            "the sidebar — only ASCII characters are allowed in header "
            "values."
        )
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
    # Surface only the exception class + first line. The full traceback
    # goes to server logs (logger.exception is called at the call site),
    # never to the UI — that's how we avoid leaking filesystem paths,
    # dependency versions, or anything else internal.
    first_line = next(iter(str(exc).splitlines()), "") or "no message"
    return (
        f"⚠️ Unexpected error ({type(exc).__name__}): `{first_line}`. "
        "Check the server logs for the traceback."
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
        logger.exception("Quiz generation failed")
        st.error(_redact_secrets(_friendly_llm_error(e)))
        return

    st.session_state["quiz"] = quiz
    st.session_state["quiz_answers"] = {}
    st.session_state["quiz_report"] = None
    st.session_state["quiz_sources"] = chunks
    st.toast(f"Generated {len(quiz.questions)} question(s).")


def _quiz_to_markdown(quiz: Quiz, *, include_answers: bool = False) -> str:
    """Render a quiz as markdown suitable for PDF export."""
    lines: list[str] = [
        f"# Quiz: {quiz.topic}",
        f"*Difficulty:* **{quiz.difficulty}** · *Questions:* **{len(quiz.questions)}**",
        "",
    ]
    for q in quiz.questions:
        lines.append(f"## Q{q.id}. {q.prompt}")
        if q.type == "multiple_choice" and q.options:
            for i, opt in enumerate(q.options):
                lines.append(f"- {chr(65 + i)}. {opt}")
        elif q.type == "true_false":
            lines.append("- True")
            lines.append("- False")
        if include_answers:
            try:
                if q.type in {"multiple_choice", "true_false"} and q.options:
                    answer_label = q.options[int(q.correct_answer)]
                else:
                    answer_label = str(q.correct_answer)
            except (TypeError, ValueError, IndexError):
                answer_label = str(q.correct_answer)
            lines.append("")
            lines.append(f"**Answer:** {answer_label}")
            if q.explanation:
                lines.append(f"**Why:** {q.explanation}")
        lines.append("")
    return "\n".join(lines)


def _grade_report_to_markdown(quiz: Quiz, report: GradeReport) -> str:
    """Render a graded quiz report as markdown suitable for PDF export."""
    lines: list[str] = [
        f"# Quiz results: {quiz.topic}",
        f"**Score:** {report.score} / {report.total} ({report.percent:.1f}%)",
        f"*Difficulty:* {quiz.difficulty}",
        "",
        "---",
        "",
    ]
    by_id = {q.id: q for q in quiz.questions}
    for r in report.per_question:
        q = by_id.get(r.id)
        if q is None:
            continue
        icon = "[OK]" if r.correct else "[X]"
        lines.append(f"## {icon} Q{q.id}. {q.prompt}")
        try:
            if q.type in {"multiple_choice", "true_false"} and q.options:
                correct_label = q.options[int(q.correct_answer)]
                user_label = (
                    q.options[int(r.user_answer)]
                    if r.user_answer is not None
                    else "(blank)"
                )
            else:
                correct_label = str(q.correct_answer)
                user_label = r.user_answer or "(blank)"
        except (TypeError, ValueError, IndexError):
            correct_label = str(q.correct_answer)
            user_label = str(r.user_answer) if r.user_answer is not None else "(blank)"
        lines.append(f"- **Correct answer:** {correct_label}")
        lines.append(f"- **Your answer:** {user_label}")
        if q.explanation:
            lines.append(f"- **Why:** {q.explanation}")
        if r.source_refs:
            lines.append(f"- **Sources:** {', '.join(r.source_refs)}")
        lines.append("")
    return "\n".join(lines)


def _render_active_quiz() -> None:
    """Render the active quiz form, grading, and retry actions."""
    quiz: Quiz = st.session_state["quiz"]
    answers: dict[int, Any] = st.session_state.setdefault("quiz_answers", {})
    report: Optional[GradeReport] = st.session_state.get("quiz_report")
    retry_only: set[int] = set(st.session_state.get("retry_ids", set()))

    st.markdown("---")
    st.markdown(f"### {quiz.topic}  · _{quiz.difficulty}_")
    _render_pdf_save_button(
        title=f"Quiz - {quiz.topic}",
        content=_quiz_to_markdown(quiz, include_answers=False),
        key=f"quiz_{id(quiz)}",
        label="📄 Save quiz as PDF",
    )

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

    _render_pdf_save_button(
        title=f"Quiz results - {quiz.topic}",
        content=_grade_report_to_markdown(quiz, report),
        key=f"report_{id(report)}",
        label="📄 Save results as PDF",
    )

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
    # Best-effort TTL sweep — runs once per process via @st.cache_resource.
    _run_startup_cleanup()
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
