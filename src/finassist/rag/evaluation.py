"""Shared evidence-evaluation contracts for domain RAG workflows."""

from typing import Protocol

from pydantic import BaseModel, Field

from finassist.rag.models import RetrievedDocument


class ContextEvaluation(BaseModel):
    sufficient: bool
    missing_concepts: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list, max_length=4)
    reason: str


class EvidenceEvaluator(Protocol):
    async def evaluate(
        self, query: str, documents: list[RetrievedDocument]
    ) -> ContextEvaluation: ...
