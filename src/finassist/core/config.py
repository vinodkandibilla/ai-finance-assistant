from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from finassist.prompts import INTENT_ROUTER_SYSTEM_PROMPT


class Settings(BaseSettings):
    portfolio_simulated_user_id: str | None = Field(default="PM001", min_length=1, max_length=100)
    portfolio_simulated_user_role: Literal["PORTFOLIO_MANAGER", "CUSTOMER"] | None = (
        "PORTFOLIO_MANAGER"
    )
    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    router_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    openai_embedding_model: str = "text-embedding-3-small"
    finance_query_analysis_model: str = "gpt-4o-mini"
    finance_evaluator_model: str = "gpt-4o-mini"
    finance_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    finance_collection: str = "finance_education_hybrid"
    finance_checkpoint_path: str = ".cache/finance_agent_checkpoints.sqlite"
    market_quotes_enabled: bool = True
    market_cache_max_entries: int = Field(default=100, gt=0)
    twelve_data_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    tavily_timeout_seconds: float = Field(default=15.0, gt=0, allow_inf_nan=False)
    twelve_data_mcp_command: str = ".cache/twelvedata-mcp/.venv/bin/python"
    twelve_data_mcp_args: list[str] = [".cache/twelvedata-mcp/src/server.py"]
    market_agent_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    market_provider_max_calls_per_request: int = Field(default=8, gt=0, le=50)
    market_provider_max_concurrency: int = Field(default=2, gt=0, le=16)
    market_provider_max_retries: int = Field(default=2, ge=0, le=3)
    market_provider_backoff_base_seconds: float = Field(default=0.5, ge=0, le=10)
    market_provider_backoff_max_seconds: float = Field(default=4.0, gt=0, le=30)
    intent_router_system_prompt: str = Field(
        default=INTENT_ROUTER_SYSTEM_PROMPT,
        min_length=1,
    )
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    @field_validator("qdrant_mode", mode="before")
    @classmethod
    def normalize_qdrant_mode(cls, value: str) -> str:
        if value == "memory":
            return "local"
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
