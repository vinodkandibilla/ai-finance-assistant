from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TaxAgentSettings(BaseSettings):
    agent_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    agent_max_searches: int = Field(default=4, ge=1, le=20)
    agent_recursion_limit: int = Field(default=30, ge=6, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)

    model_config = SettingsConfigDict(env_prefix="TAX_", extra="ignore")
