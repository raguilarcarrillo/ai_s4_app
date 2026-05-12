"""Pluggable LLM provider abstraction for Smart Teacher.

This module exposes a single :func:`get_llm` entry point that returns a
LangChain ``BaseChatModel`` for any supported provider. The rest of the
application is provider-unaware — adding a new provider only requires editing
this file.

Supported providers:
    * ``anthropic`` — Anthropic Claude (langchain-anthropic)
    * ``openai`` — OpenAI GPT (langchain-openai)
    * ``google`` — Google Gemini (langchain-google-genai)
    * ``groq`` — Groq Cloud (langchain-groq)
    * ``huggingface`` — HuggingFace Inference API (langchain-huggingface)
    * ``ollama`` — Local Ollama runtime (langchain-ollama)

Adding a new provider (recipe):
    1. Add an entry to :data:`PROVIDERS` with default model + env var name.
    2. Add a branch to :func:`get_llm` that imports the provider package
       lazily and instantiates the chat model.
    3. (Optional) Update the README's provider table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProviderSpec:
    """Static metadata about a supported LLM provider.

    Attributes:
        key: Internal provider id used by the rest of the app (e.g. ``openai``).
        label: Human-friendly name shown in the UI.
        default_model: Sensible default model identifier for the provider.
        env_var: Environment variable / Streamlit secret that holds the API
            key. Empty string for providers that don't require one (Ollama).
        needs_key: Whether the provider needs an API key at all.
        notes: Short descriptor shown in the sidebar.
    """

    key: str
    label: str
    default_model: str
    env_var: str
    needs_key: bool
    notes: str


PROVIDERS: Dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic Claude",
        default_model="claude-sonnet-4-6",
        env_var="ANTHROPIC_API_KEY",
        needs_key=True,
        notes="High-quality reasoning; paid API.",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI GPT",
        default_model="gpt-4o-mini",
        env_var="OPENAI_API_KEY",
        needs_key=True,
        notes="Widely supported; paid API.",
    ),
    "google": ProviderSpec(
        key="google",
        label="Google Gemini",
        default_model="gemini-2.5-flash",
        env_var="GOOGLE_API_KEY",
        needs_key=True,
        notes="Generous free tier on Gemini Flash.",
    ),
    "groq": ProviderSpec(
        key="groq",
        label="Groq",
        default_model="llama-3.3-70b-versatile",
        env_var="GROQ_API_KEY",
        needs_key=True,
        notes="Very fast inference; free developer tier.",
    ),
    "huggingface": ProviderSpec(
        key="huggingface",
        label="HuggingFace Inference",
        default_model="meta-llama/Meta-Llama-3-8B-Instruct",
        env_var="HUGGINGFACEHUB_API_TOKEN",
        needs_key=True,
        notes="Free tier with rate limits.",
    ),
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (local)",
        default_model="llama3.2",
        env_var="",
        needs_key=False,
        notes="Local models; needs Ollama running.",
    ),
}


def list_providers() -> Dict[str, ProviderSpec]:
    """Return the table of supported providers.

    Returns:
        Mapping of provider key -> :class:`ProviderSpec`.
    """
    return PROVIDERS


def resolve_api_key(provider: str, user_key: Optional[str] = None) -> Optional[str]:
    """Resolve an API key for ``provider``.

    Resolution order: explicit ``user_key`` (from the sidebar) >
    ``os.environ`` > ``st.secrets``. Returns ``None`` if the provider does not
    require a key.

    Args:
        provider: Provider key, e.g. ``"anthropic"``.
        user_key: Optional key entered by the user in the UI. Wins if truthy.

    Returns:
        The resolved key, or ``None`` if the provider needs none / nothing
        was found.

    Raises:
        ValueError: If ``provider`` is unknown.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}")
    spec = PROVIDERS[provider]
    if not spec.needs_key:
        return None

    if user_key:
        return user_key.strip()

    env_val = os.environ.get(spec.env_var)
    if env_val:
        return env_val.strip()

    # Streamlit secrets are optional — only check if running under Streamlit.
    try:
        import streamlit as st  # type: ignore

        if spec.env_var in st.secrets:
            return str(st.secrets[spec.env_var]).strip()
    except Exception:
        # Either Streamlit isn't installed or secrets isn't configured.
        pass
    return None


def get_llm(
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a chat LLM for the requested provider.

    The returned object implements the LangChain ``BaseChatModel`` interface,
    so callers don't need to know which provider is in use.

    Args:
        provider: One of the keys in :data:`PROVIDERS`.
        model: Model identifier. Falls back to the provider's default.
        temperature: Sampling temperature in ``[0.0, 2.0]``.
        api_key: User-supplied key. If absent, resolved from env/secrets.
        **kwargs: Extra provider-specific kwargs passed through (e.g.
            ``base_url`` for Ollama, ``max_tokens``, ``top_p``).

    Returns:
        A configured chat model instance.

    Raises:
        ValueError: If the provider is unknown or a required key is missing.
        ImportError: If the provider package isn't installed.

    Example:
        >>> llm = get_llm("groq", model="llama-3.3-70b-versatile",
        ...               temperature=0.0, api_key="gsk_...")
        >>> response = llm.invoke("Explain photosynthesis in one sentence.")
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {list(PROVIDERS)}"
        )
    spec = PROVIDERS[provider]
    model = model or spec.default_model
    resolved_key = resolve_api_key(provider, api_key)
    if spec.needs_key and not resolved_key:
        raise ValueError(
            f"Missing API key for {spec.label}. Set {spec.env_var} or enter "
            f"the key in the sidebar."
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=resolved_key,
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=resolved_key,
            **kwargs,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=resolved_key,
            **kwargs,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=resolved_key,
            **kwargs,
        )

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            temperature=max(temperature, 0.01),  # HF endpoint dislikes 0.
            huggingfacehub_api_token=resolved_key,
            **kwargs,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = kwargs.pop(
            "base_url",
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            **kwargs,
        )

    # Unreachable thanks to the membership check above.
    raise ValueError(f"Unhandled provider: {provider!r}")
