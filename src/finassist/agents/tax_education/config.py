from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TaxAgentSettings(BaseSettings):
    agent_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    agent_max_searches: int = Field(default=4, ge=1, le=20)
    agent_recursion_limit: int = Field(default=30, ge=6, le=100)
    reranker_enabled: bool = False
    answer_cache_enabled: bool = True
    answer_cache_collection: str = "tax_answer_cache"
    answer_cache_ttl_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    answer_cache_similarity: float = Field(default=0.90, gt=0, le=1, allow_inf_nan=False)
    answer_cache_embedding_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    expansion_rerank_top_n: int = Field(default=10, ge=1, le=20)
    expansion_search_limit: int = Field(default=40, ge=1, le=100)
    max_expansion_actions: int = Field(default=1, ge=0, le=10)
    max_targeted_actions: int = Field(default=2, ge=0, le=10)
    max_evidence_passages: int = Field(default=15, ge=1, le=100)
    max_evidence_characters: int = Field(default=30000, ge=1)

    model_config = SettingsConfigDict(env_prefix="TAX_", env_file=".env", extra="ignore")
