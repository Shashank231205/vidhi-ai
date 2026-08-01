"""Typed application settings.

All configuration is environment-driven. There are no local-filesystem defaults:
the database, cache, and model endpoints are hosted services in every environment,
including local development.
"""

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """Providers are OpenAI-compatible, so one client covers all of them."""

    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    #: Browser origins allowed to call the API. The deployed frontend origin is
    #: appended here at deploy time; the defaults cover local Next.js dev.
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"])

    # --- Data layer (Supabase) ---
    database_url: PostgresDsn = Field(
        description="Supabase Postgres connection string (pooler endpoint)."
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_command_timeout_s: float = 30.0

    # --- Cache (Upstash Redis over HTTP) ---
    upstash_redis_url: str = Field(description="Upstash REST endpoint.")
    upstash_redis_token: SecretStr

    embedding_cache_ttl_s: int = 60 * 60 * 24 * 30  # content-addressed, safe to keep
    prediction_cache_ttl_s: int = 60 * 60 * 24 * 7
    query_cache_ttl_s: int = 60 * 15

    # --- LLM routing ---
    llm_provider_order: list[LLMProvider] = Field(
        default=[LLMProvider.GROQ, LLMProvider.CEREBRAS, LLMProvider.OPENROUTER],
        description="Failover order. Exhausted left to right on rate-limit or error.",
    )
    groq_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    llm_request_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- Embeddings + classifiers (HuggingFace Inference API) ---
    hf_api_token: SecretStr
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 32

    risk_classifier_model: str | None = None  # set in Phase 6
    stance_classifier_model: str | None = None

    # --- Agent behaviour ---
    max_retrieval_attempts: int = Field(
        default=3, ge=1, le=5, description="Critic-loop reformulation cap."
    )
    max_grounding_attempts: int = Field(
        default=2, ge=1, le=4, description="Verifier reject-and-retry cap."
    )
    retrieval_top_k: int = 8
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    @field_validator("llm_provider_order")
    @classmethod
    def _non_empty(cls, v: list[LLMProvider]) -> list[LLMProvider]:
        if not v:
            raise ValueError("llm_provider_order must list at least one provider")
        return v

    @property
    def is_local(self) -> bool:
        return self.environment is Environment.LOCAL

    def api_key_for(self, provider: LLMProvider) -> SecretStr | None:
        return {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.CEREBRAS: self.cerebras_api_key,
            LLMProvider.OPENROUTER: self.openrouter_api_key,
        }[provider]

    def configured_providers(self) -> list[LLMProvider]:
        """Providers that are both enabled and hold a usable key."""
        return [p for p in self.llm_provider_order if self.api_key_for(p) is not None]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
