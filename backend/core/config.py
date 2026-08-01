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


class EmbeddingBackend(StrEnum):
    """Where embeddings are computed.

    LOCAL runs the model in-process: ~30ms per query versus ~500ms-1.3s via the
    hosted API, with no cold starts or rate limits. REMOTE keeps the process
    small, for deployment targets where a 2GB model is not welcome. Both use
    the same weights, so a corpus embedded by one is valid for the other.
    """

    LOCAL = "local"
    REMOTE = "remote"


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
        # OpenRouter before Cerebras: Cerebras' free tier now requires billing
        # and answers 402, so keeping it second would put a dead hop between
        # two working providers on every failover.
        default=[LLMProvider.GROQ, LLMProvider.OPENROUTER, LLMProvider.CEREBRAS],
        description="Failover order. Exhausted left to right on rate-limit or error.",
    )
    groq_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    llm_request_timeout_s: float = 60.0
    llm_max_retries: int = 2
    #: Longest a 429's retry-after is honoured before failing over. Groq's free
    #: tier recovers in seconds and is much faster than the fallbacks, so a
    #: short wait beats switching; a long one does not.
    llm_max_retry_wait_s: float = Field(default=8.0, ge=0.0, le=60.0)
    #: Per-provider model override, keyed by provider value. Lets a model be
    #: swapped without a code change when a provider retires one.
    llm_model_overrides: dict[str, str] = Field(default_factory=dict)

    # --- Embeddings + classifiers (HuggingFace Inference API) ---
    embedding_backend: EmbeddingBackend = EmbeddingBackend.LOCAL
    #: Only required when embedding_backend is REMOTE.
    hf_api_token: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 32

    #: Local path or HF repo id. Unset means the LLM path is used, which is
    #: also the baseline the classifier is measured against.
    risk_classifier_model: str | None = None
    stance_classifier_model: str | None = None
    #: Below this, the LLM's reasoned level stands: it at least saw the
    #: statutory context, which the classifier does not.
    risk_classifier_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    stance_classifier_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    # --- Agent behaviour ---
    max_retrieval_attempts: int = Field(
        default=3, ge=1, le=5, description="Critic-loop reformulation cap."
    )
    max_grounding_attempts: int = Field(
        default=2, ge=1, le=4, description="Verifier reject-and-retry cap."
    )
    #: How many clauses are audited at once. Clauses are independent, so this
    #: is the main lever on audit latency; capped to stay inside provider rate
    #: limits rather than to protect our own resources.
    max_concurrent_clauses: int = Field(default=6, ge=1, le=20)

    retrieval_top_k: int = 8
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    #: Fusion weights, chosen by sweeping both against the DPDP eval set rather
    #: than by intuition. On this corpus BGE-M3 dominates: equal weighting
    #: scored *below* vector alone (MRR 0.633 vs 0.842), because the lexical
    #: arm's weak tail outvoted confident vector hits. A small lexical
    #: contribution over only its top few results is the measured optimum —
    #: recall@5 rises 0.875 → 0.925 for ~0.01 MRR, i.e. it surfaces relevant
    #: sections the vector arm misses without disturbing the ranking.
    #:
    #: These are corpus-dependent. Re-run eval/retrieval_eval.py after adding a
    #: statute or changing the embedding model.
    rrf_vector_weight: float = 1.0
    rrf_lexical_weight: float = 0.1
    #: Depth cap for the lexical arm; past its top few, OR-matched results
    #: share only common words and add noise.
    lexical_candidate_limit: int = 3

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
