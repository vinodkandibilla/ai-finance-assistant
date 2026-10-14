"""Evidence returned by news search, without generated synthesis."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from finassist.agents.common.schemas import AgentResult


class NewsEvidence(BaseModel):
    evidence_id: str = Field(pattern=r"^E[1-9]\d*$")
    title: str = Field(min_length=1)
    source: str | None = None
    url: str
    published_at: str | None = None
    content: str = Field(min_length=1)

    @field_validator("title", "content")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence text must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def require_web_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Evidence requires an HTTP(S) URL")
        return value  # Preserve the provider's exact URL, including query parameters.


class NewsClaim(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    kind: Literal["reported_fact", "interpretation"] = "reported_fact"


class NewsAgentResult(AgentResult):
    claims: list[NewsClaim] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    next_search_query: str | None = None
