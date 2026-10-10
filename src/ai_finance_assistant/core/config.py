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
    finance_query_analysis_model: str = "gpt-4o-mini"
    finance_evaluator_model: str = "gpt-4o-mini"
    finance_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    finance_collection: str = "finance_education_dense"
    finance_checkpoint_path: str = ".cache/finance_agent_checkpoints.sqlite"
    alpha_vantage_api_key: SecretStr | None = None
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    alpha_vantage_timeout_seconds: float = 3.0
    quote_cache_ttl_seconds: int = 24 * 60 * 60
    intent_router_system_prompt: str = Field(
        default=(
            "Classify the user's request by intent. Return finance_qa for financial education "
            "questions about financial concepts, products, investing, or options education; "
            "return portfolio for requests to analyze "
            "portfolio holdings, allocation, concentration, or performance; return market "
            "for current stock quotes on MAG7 tickers (AAPL, MSFT, AMZN, GOOG, GOOGL, META, "
            "NVDA, TSLA); return clarify for historical prices, volume, comparisons, or "
            "unsupported tickers; return goals "
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
