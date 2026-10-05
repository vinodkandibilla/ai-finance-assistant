from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    intent_router_system_prompt: str = Field(
        default=(
            "Classify the user's request by intent. Return finance_qa for questions about "
            "financial concepts or products; return portfolio for requests to analyze "
            "portfolio holdings, allocation, concentration, or performance; return market "
            "for current or historical security prices, index levels, trading volume, "
            "price trends, or comparisons between securities or benchmarks; return goals "
            "for savings targets, future-value projections, required contributions, "
            "timelines, or financial-planning calculations; return news for current AI, "
            "tech, market, or industry news summaries, headline roundups, and brief "
            "breaking-news updates; return tax for IRS guidance, tax education, filing "
            "status, deductions, credits, 401(k), traditional IRA, Roth IRA, and related "
            "tax-planning questions; return clarify when the query is too vague or "
            "missing essential details required to answer accurately, such as a stock "
            "ticker, date range, filing status, child count, account type, or topic "
            "scope. Return other for requests outside those categories. Do not answer the "
            "question."
        ),
        min_length=1,
    )
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "finance_knowledge"

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
