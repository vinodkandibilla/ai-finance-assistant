from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from finassist.rag.models import RetrievedDocument

DomainIntent = Literal["finance_qa", "tax", "market", "news", "portfolio", "goals", "other"]


class AgentResult(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    status: Literal["answered", "clarification", "out_of_scope", "insufficient", "error"] = Field(
        description=(
            "out_of_scope: the user's task is outside this agent's responsibility; "
            "insufficient: the task is in scope but evidence is missing; "
            "answered: evidence-supported completion; error: execution failure."
        )
    )
    clarification_question: str | None = None
    errors: list[str] = Field(default_factory=list)


class EvidenceResult(BaseModel):
    candidates: list[RetrievedDocument] = Field(default_factory=list)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    query: str
    user_query: str
    search_query: str
    intent: str
    answer: str
    status: str
    citations: list[str]
    clarification_question: str
    calculations: list[dict]
    calculation: dict
    final_result: dict
    errors: list[str]
