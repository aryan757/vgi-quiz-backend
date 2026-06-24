"""Application configuration, loaded from environment / .env via pydantic-settings.

Every tunable lives here so the rest of the codebase never reads os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Mongo ---
    database_url: str = Field(..., alias="DATABASE_URL")
    database_name: str = Field("vgi_skill_lab", alias="DATABASE_NAME")
    knowledge_base_collection: str = Field("knowledge_base", alias="KNOWLEDGE_BASE_COLLECTION")
    question_bank_collection: str = Field("question_bank", alias="QUESTION_BANK_COLLECTION")

    # --- LLM provider (generation only) ---
    llm_provider: Literal["anthropic", "openai"] = Field("openai", alias="LLM_PROVIDER")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    llm_model_name: str = Field("gpt-4o-mini", alias="LLM_MODEL_NAME")

    # --- OpenAI (always required: embeddings are OpenAI-only) ---
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")

    # --- Topic matching ---
    topic_match_threshold: float = Field(0.80, alias="TOPIC_MATCH_THRESHOLD")

    # --- Seeding ---
    seed_questions_per_topic_per_difficulty: int = Field(
        100, alias="SEED_QUESTIONS_PER_TOPIC_PER_DIFFICULTY"
    )
    seed_duplicate_threshold: float = Field(0.95, alias="SEED_DUPLICATE_THRESHOLD")

    # --- App ---
    app_env: str = Field("development", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def llm_api_key(self) -> str:
        """The API key for the *generation* provider currently selected."""
        return self.anthropic_api_key if self.llm_provider == "anthropic" else self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Import this everywhere instead of re-reading env."""
    return Settings()  # type: ignore[call-arg]
