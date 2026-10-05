from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    content: str
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)
    score: float | None = None
