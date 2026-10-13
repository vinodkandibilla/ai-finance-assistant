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
    market_cache_max_entries: int = Field(default=100, gt=0)
    twelve_data_api_key: SecretStr | None = None
    twelve_data_mcp_command: str = ".cache/twelvedata-mcp/.venv/bin/python"
    twelve_data_mcp_args: list[str] = [".cache/twelvedata-mcp/src/server.py"]
    market_agent_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
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
            "market: current prices, quotes, volume, and daily change for stocks, ETFs "
            "(including AAPL and QQQ), forex, or crypto; company names and missing or "
            "ambiguous symbols are resolved by the market agent. "
            "Historical prices and time-series windows, including weekly/daily data and "
            "calendar or fiscal quarters, also route to market; that agent owns timeframe "
            "clarification. Single-asset price growth, returns, or percentage changes over "
            "a horizon and cross-asset price-performance comparisons also route to market. "
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
