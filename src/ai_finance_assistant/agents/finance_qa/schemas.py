from __future__ import annotations

from pydantic import BaseModel, Field

from ai_finance_assistant.agents.common.schemas import AgentResult, DomainIntent
from ai_finance_assistant.rag.evaluation import ContextEvaluation as ContextEvaluation
from ai_finance_assistant.rag.models import RetrievedDocument

FinanceDomainIntent = DomainIntent


class QueryAnalysis(BaseModel):
    needs_clarification: bool
    clarification_question: str | None = None
    detected_intents: list[FinanceDomainIntent] = Field(default_factory=list)
    is_complex: bool = False
    sparse_keywords: list[str] = Field(default_factory=list)
    dense_sub_queries: list[str] = Field(default_factory=list, max_length=3)


FinanceDocument = RetrievedDocument


class FinanceAgentResult(AgentResult):
    """Finance education result returned to the API."""
