from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    content: str
    source: str
    metadata: dict[str, str | int | float | bool | list[str] | None] = Field(default_factory=dict)
    score: float | None = None
    doc_id: str | None = None
    rank: int | None = None
    provenance: list[str] = Field(default_factory=list)
