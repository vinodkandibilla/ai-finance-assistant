from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FINANCE_PDF_PATH = (
    PROJECT_ROOT
    / "src"
    / "ai_finance_assistant"
    / "rag"
    / "finance-qa"
    / "Financial_Education_Handbook_integrated.pdf"
)


class FinanceAgentSettings(BaseSettings):
    dense_top_k: int = Field(default=10, ge=1, le=100)
    sparse_top_k: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    rrf_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    max_retrieval_retries: int = Field(default=1, ge=0, le=1)
    max_parallel_searches: int = Field(default=4, ge=1, le=16)
    retrieval_timeout_seconds: float = Field(default=15.0, gt=0)
    query_analysis_model: str = "gpt-4o-mini"
    evaluator_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    finance_collection: str = "finance_education_dense"
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    sufficiency_tie_tolerance: float = Field(default=0.08, ge=0, le=1)

    model_config = SettingsConfigDict(env_prefix="FINANCE_", extra="ignore")