from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.graph.builder import FinanceAssistantGraph, build_graph
from ai_finance_assistant.graph.classifier import IntentClassifier, create_intent_classifier
from ai_finance_assistant.graph.state import FinanceIntent
from ai_finance_assistant.providers.market_data import get_stock_price
from ai_finance_assistant.rag.finance_education import get_finance_retriever
from ai_finance_assistant.rag.tax_education import get_tax_retriever

router = APIRouter(tags=["assistant"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class QueryResponse(BaseModel):
    intent: FinanceIntent
    answer: str


@lru_cache
def get_query_graph() -> FinanceAssistantGraph:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The intent classifier is not configured. Set OPENAI_API_KEY.",
        )
    classifier: IntentClassifier = create_intent_classifier(settings)
    retriever = None
    tax_retriever = None
    llm = None
    if settings.openai_api_key is not None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        retriever = get_finance_retriever()
        qdrant_client = getattr(retriever.vector_store, "client", None)
        tax_retriever = get_tax_retriever(qdrant_client)
    return build_graph(
        classifier,
        retriever=retriever,
        tax_retriever=tax_retriever,
        llm=llm,
        market_quote_tool=get_stock_price,
    )


@router.post("/query", response_model=QueryResponse)
async def query_assistant(
    request: QueryRequest,
    graph: Annotated[FinanceAssistantGraph, Depends(get_query_graph)],
) -> QueryResponse:
    result = await graph.ainvoke({"query": request.query})
    return QueryResponse(intent=result["intent"], answer=result["answer"])
