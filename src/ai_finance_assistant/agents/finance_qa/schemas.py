from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FinanceDomainIntent = Literal["finance_qa", "tax", "market", "news", "portfolio", "goals", "other"]


class QueryAnalysis(BaseModel):
    needs_clarification: bool
    clarification_question: str | None = None
    detected_intents: list[FinanceDomainIntent] = Field(default_factory=list)
    is_complex: bool = False
    sparse_keywords: list[str] = Field(default_factory=list)
    dense_sub_queries: list[str] = Field(default_factory=list, max_length=3)


class ContextEvaluation(BaseModel):
    sufficient: bool
    sufficiency_score: float = Field(ge=0, le=1)
    missing_concepts: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list, max_length=4)
    reason: str


class FinanceDocument(BaseModel):
    doc_id: str
    content: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    source: str
    score: float = 0.0
    rank: int = 0
    provenance: list[str] = Field(default_factory=list)


class FinanceAgentResult(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    status: Literal["answered", "clarification", "handoff", "insufficient", "error"]
    handoff_intents: list[FinanceDomainIntent] = Field(default_factory=list)
    clarification_question: str | None = None
    errors: list[str] = Field(default_factory=list)