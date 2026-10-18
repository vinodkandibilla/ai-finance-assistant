"""Frontend-owned wire models matching finassist-openapi.json."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

Intent = Literal["finance_qa", "portfolio", "market", "goals", "news", "tax", "clarify", "other"]
Handler = Literal[
    "finance_qa_node",
    "tax_education_node",
    "market_analysis_node",
    "portfolio_analysis_node",
    "goals_node",
    "news_synthesizer_node",
    "clarify_node",
    "unsupported_node",
]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Identifier = Annotated[str, Field(min_length=1, max_length=100)]


class WireModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class HealthResponse(WireModel):
    status: Literal["ok"]
    version: str


class QueryRequest(WireModel):
    query: Question
    user_role: Literal["portfolio_manager", "individual_customer"] | None = None
    customer_id: Identifier | None = None
    manager_id: Identifier | None = None


class ResumeRequest(WireModel):
    thread_id: str = Field(min_length=1)
    reply: Question


class QueryResponse(WireModel):
    intent: Intent
    answer: str
    classified_intent: Intent | None = None
    executed_handler: Handler | None = None
    calculation: dict[str, JsonValue] | None = None
    calculations: list[dict[str, JsonValue]] | None = None
    status: str | None = None
    citations: list[str] | None = None
    thread_id: str | None = None
    clarification_question: str | None = None
