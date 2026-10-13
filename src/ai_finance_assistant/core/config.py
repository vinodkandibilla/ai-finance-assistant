from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    alpha_vantage_api_key: SecretStr | None = None
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    alpha_vantage_timeout_seconds: float = 3.0
    quote_cache_ttl_seconds: int = 24 * 60 * 60
    intent_router_system_prompt: str = Field(
        default=(
            "Choose exactly one intent by the requested task, in the priority order below; "
            "do not answer. Incidental tax, retirement, deduction, or account mentions "
            "do not determine intent. "
            "tax: IRS/tax education, treatment, filing status, deductions, credits, planning, "
            "or retirement-account rules (Roth/Traditional/SEP/SIMPLE IRA, 401(k), 403(b), "
            "similar plans), including eligibility, contributions/limits, withdrawals/"
            "distributions, conversions, definitions, and comparisons of those rules. "
            "finance_qa: general investing concepts, securities, funds/ETFs, benchmarks, "
            "options mechanics, compounding explanations and single-principal "
            "simple/compound-interest "
            "calculations, diversification, and risk; tax/retirement "
            "background alone does not exclude it. "
            "portfolio: analysis of personal holdings, allocation, concentration, or performance. "
            "market: current quotes for AAPL/MSFT/AMZN/GOOG/GOOGL/META/NVDA/TSLA; "
            "market requests for historical prices, volume, comparisons, unsupported tickers, "
            "or missing symbols require clarify. "
            "goals: savings targets, future values, required contributions, timelines, or "
            "planning calculations, except tax-advantaged account rules and single-principal "
            "simple/compound-interest calculations (finance_qa). "
            "news: current AI/tech/market/industry news summaries or breaking news. "
            "clarify: domain unclear (e.g. 'Which is better?') or the market cases above. "
            "Identifiable tax/finance_qa requests stay in that domain despite missing "
            "income, filing status, year, comparands, or other details; their agents clarify. "
            "Examples: 'What will be my tax bracket?' -> tax; 'Compare two ETFs' -> finance_qa. "
            "other: outside these categories."
        ),
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
