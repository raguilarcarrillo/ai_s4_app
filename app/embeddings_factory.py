"""Pluggable embedding provider abstraction for Smart Teacher.

Mirrors the design of :mod:`llm_factory`: a single :func:`get_embeddings`
entry point returns a LangChain ``Embeddings`` instance for any supported
backend. The default backend is local Sentence-Transformers, which works
without any API key and is free on HuggingFace Spaces.

Supported backends:
    * ``sentence-transformers`` (default) — local CPU embeddings, no key.
    * ``huggingface`` — HuggingFace Inference API, requires a token.
    * ``openai`` — OpenAI ``text-embedding-3-*``, requires an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EmbeddingSpec:
    """Metadata for a supported embedding backend.

    Attributes:
        key: Internal id of the backend.
        label: Human-friendly name.
        default_model: Default model identifier.
        env_var: Env var holding the API key (empty if none required).
        needs_key: Whether a key is required.
    """

    key: str
    label: str
    default_model: str
    env_var: str
    needs_key: bool


EMBEDDINGS: Dict[str, EmbeddingSpec] = {
    "sentence-transformers": EmbeddingSpec(
        key="sentence-transformers",
        label="Sentence-Transformers (local, free)",
        default_model="sentence-transformers/all-MiniLM-L6-v2",
        env_var="",
        needs_key=False,
    ),
    "huggingface": EmbeddingSpec(
        key="huggingface",
        label="HuggingFace Inference API",
        default_model="sentence-transformers/all-MiniLM-L6-v2",
        env_var="HUGGINGFACEHUB_API_TOKEN",
        needs_key=True,
    ),
    "openai": EmbeddingSpec(
        key="openai",
        label="OpenAI Embeddings",
        default_model="text-embedding-3-small",
        env_var="OPENAI_API_KEY",
        needs_key=True,
    ),
}


def list_embedding_backends() -> Dict[str, EmbeddingSpec]:
    """Return the table of supported embedding backends.

    Returns:
        Mapping of backend key -> :class:`EmbeddingSpec`.
    """
    return EMBEDDINGS


def _resolve_key(env_var: str, user_key: Optional[str]) -> Optional[str]:
    """Resolve an API key from sidebar > env > Streamlit secrets.

    Args:
        env_var: Name of the env var holding the key.
        user_key: Optional key entered in the UI.

    Returns:
        The resolved key, or ``None`` if not found.
    """
    if user_key:
        return user_key.strip()
    if not env_var:
        return None
    value = os.environ.get(env_var)
    if value:
        return value.strip()
    try:
        import streamlit as st  # type: ignore

        if env_var in st.secrets:
            return str(st.secrets[env_var]).strip()
    except Exception:
        pass
    return None


def get_embeddings(
    backend: str = "sentence-transformers",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate an embedding model for the requested backend.

    Args:
        backend: One of the keys in :data:`EMBEDDINGS`.
        model: Model identifier; falls back to the backend default.
        api_key: User-supplied key for cloud backends.
        **kwargs: Extra backend-specific kwargs.

    Returns:
        A LangChain ``Embeddings`` instance.

    Raises:
        ValueError: If the backend is unknown or a required key is missing.
        ImportError: If the backend package isn't installed.

    Example:
        >>> emb = get_embeddings()  # local sentence-transformers
        >>> vec = emb.embed_query("photosynthesis")
    """
    if backend not in EMBEDDINGS:
        raise ValueError(
            f"Unknown embeddings backend {backend!r}. "
            f"Supported: {list(EMBEDDINGS)}"
        )
    spec = EMBEDDINGS[backend]
    model = model or spec.default_model
    key = _resolve_key(spec.env_var, api_key) if spec.needs_key else None
    if spec.needs_key and not key:
        raise ValueError(
            f"Missing API key for {spec.label}. Set {spec.env_var} or enter "
            f"the key in the sidebar."
        )

    if backend == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": kwargs.pop("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True, **kwargs},
        )

    if backend == "huggingface":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=key,
            **kwargs,
        )

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=key, **kwargs)

    raise ValueError(f"Unhandled embeddings backend: {backend!r}")
