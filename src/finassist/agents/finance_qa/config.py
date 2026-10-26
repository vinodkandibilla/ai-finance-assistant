from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FinanceAgentSettings(BaseSettings):
    agent_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    agent_max_searches: int = Field(default=4, ge=1, le=20)
    agent_recursion_limit: int = Field(default=30, ge=6, le=100)
    dense_top_k: int = Field(default=10, ge=1, le=100)
    sparse_top_k: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    rrf_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=3, ge=1, le=20)
    expansion_dense_top_k: int = Field(default=20, ge=1, le=100)
    expansion_sparse_top_k: int = Field(default=20, ge=1, le=100)
    expansion_rrf_candidates: int = Field(default=40, ge=1, le=100)
    expansion_rerank_top_n: int = Field(default=5, ge=1, le=20)
    max_expansion_actions: int = Field(default=1, ge=0, le=10)
    max_targeted_actions: int = Field(default=2, ge=0, le=10)
    max_dense_queries: int = Field(default=3, ge=1, le=3)
    max_evidence_passages: int = Field(default=15, ge=1, le=100)
    max_evidence_characters: int = Field(default=30000, ge=1)
    max_retrieval_retries: int = Field(default=1, ge=0, le=1)
    max_parallel_searches: int = Field(default=4, ge=1, le=16)
    retrieval_timeout_seconds: float = Field(default=15.0, gt=0)
    query_analysis_model: str = "gpt-4o-mini"
    evaluator_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = False
    answer_cache_enabled: bool = True
    answer_cache_collection: str = "finance_answer_cache"
    answer_cache_ttl_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    answer_cache_similarity: float = Field(default=0.92, gt=0, le=1, allow_inf_nan=False)
    answer_cache_embedding_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    finance_collection: str = "finance_education_hybrid"
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)

    model_config = SettingsConfigDict(env_prefix="FINANCE_", env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_retrieval_profiles(self):
        if self.rerank_top_n > self.rrf_candidates:
            raise ValueError("Baseline rerank_top_n cannot exceed rrf_candidates.")
        if self.expansion_rerank_top_n > self.expansion_rrf_candidates:
            raise ValueError("Expansion rerank limit cannot exceed its candidate limit.")
        if (
            self.expansion_dense_top_k < self.dense_top_k
            or self.expansion_sparse_top_k < self.sparse_top_k
            or self.expansion_rrf_candidates < self.rrf_candidates
            or self.expansion_rerank_top_n < self.rerank_top_n
        ):
            raise ValueError("Expansion profile cannot be shallower than baseline.")
        return self
