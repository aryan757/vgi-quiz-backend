"""Provider-agnostic LLM access.

A single place that returns a configured LangChain chat model for whichever provider
(`anthropic` or `openai`) is selected via env. Used by both the seeding script and the
runtime question generator, so neither needs to know which provider is active.

Embeddings are deliberately NOT here — they are always OpenAI (see services/embeddings.py).
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_chat_model(settings: Settings | None = None, *, temperature: float = 0.4) -> BaseChatModel:
    """Build a LangChain chat model for the configured provider.

    Raises a clear error if the matching API key is missing rather than failing deep
    inside a request.
    """
    settings = settings or get_settings()

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty. Set it in .env."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model_name,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=4096,
            timeout=60,
            max_retries=1,
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. Set it in .env."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            temperature=temperature,
            timeout=60,
            max_retries=1,
        )

    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
