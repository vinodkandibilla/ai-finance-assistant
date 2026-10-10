from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, field_validator

from ai_finance_assistant.agents.finance_qa import FinanceQaAgent
from ai_finance_assistant.agents.finance_qa.config import FinanceAgentSettings
from ai_finance_assistant.agents.finance_qa.corpus_index import get_finance_corpus
from ai_finance_assistant.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from ai_finance_assistant.agents.finance_qa.reranker import FinanceCrossEncoderReranker
from ai_finance_assistant.agents.finance_qa.retrieval import FinanceHybridRetriever
from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.graph.builder import FinanceAssistantGraph, build_graph
from ai_finance_assistant.graph.classifier import IntentClassifier, create_intent_classifier
from ai_finance_assistant.graph.state import FinanceIntent
from ai_finance_assistant.providers.market_data import get_stock_price
from ai_finance_assistant.rag.qdrant import get_qdrant_client
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
    status: str | None = None
    citations: list[str] | None = None
    thread_id: str | None = None
    handoff_intents: list[str] | None = None
    clarification_question: str | None = None


async def get_query_checkpointer(request: Request) -> AsyncSqliteSaver:
    return request.app.state.query_checkpointer


@lru_cache
def get_query_graph(
    checkpointer: Annotated[AsyncSqliteSaver | None, Depends(get_query_checkpointer)] = None,
):
    settings = get_settings()
    if settings.openai_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The intent classifier is not configured. Set OPENAI_API_KEY.",
        )
    classifier: IntentClassifier = create_intent_classifier(settings)
    tax_retriever = None
    finance_agent = None
    llm = None
    if settings.openai_api_key is not None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        qdrant_client = get_qdrant_client()
        tax_retriever = get_tax_retriever(qdrant_client)

        finance_settings = FinanceAgentSettings(
            query_analysis_model=settings.finance_query_analysis_model,
            evaluator_model=settings.finance_evaluator_model,
            reranker_model=settings.finance_reranker_model,
            finance_collection=settings.finance_collection,
        )
        finance_corpus = get_finance_corpus()
        finance_metadata = FinanceMetadataExtractor(
            [document.metadata for document in finance_corpus]
        )
        finance_retriever = FinanceHybridRetriever.create(
            finance_corpus,
            qdrant_client=qdrant_client,
            api_key=settings.openai_api_key.get_secret_value(),
            embedding_model=settings.openai_embedding_model,
            settings=finance_settings,
        )
        finance_llm = ChatOpenAI(
            model=finance_settings.query_analysis_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        finance_agent = FinanceQaAgent(
            llm=finance_llm,
            retriever=finance_retriever,
            metadata_extractor=finance_metadata,
            classifier=classifier,
            reranker=FinanceCrossEncoderReranker(finance_settings.reranker_model),
            settings=finance_settings,
        )
    return build_graph(
        classifier,
        tax_retriever=tax_retriever,
        llm=llm,
        market_quote_tool=get_stock_price,
        finance_agent=finance_agent,
        checkpointer=checkpointer,
    )


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_assistant(
    request: QueryRequest,
    graph: Annotated[FinanceAssistantGraph, Depends(get_query_graph)],
) -> QueryResponse:
    import uuid

    thread_id = str(uuid.uuid4())
    result = await graph.ainvoke(
        {"query": request.query, "thread_id": thread_id},
        config={"configurable": {"thread_id": thread_id}},
    )
    interrupt_payload = await _get_interrupt_payload(graph, thread_id, result)
    status = "clarification" if interrupt_payload else result.get("status")
    if status in {None, "answered", "not_implemented"}:
        status = None
    return QueryResponse(
        intent=result["intent"],
        answer=result.get("answer", ""),
        status=status,
        citations=result.get("citations") or None,
        thread_id=thread_id if interrupt_payload else None,
        handoff_intents=result.get("handoff_intents") or None,
        clarification_question=(
            interrupt_payload.get("question")
            if interrupt_payload
            else result.get("clarification_question") or None
        ),
    )


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    reply: str = Field(min_length=1, max_length=4000)

    @field_validator("reply", mode="before")
    @classmethod
    def strip_reply(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


@router.post("/query/resume", response_model=QueryResponse, response_model_exclude_none=True)
async def resume_query(
    request: ResumeRequest,
    graph: Annotated[FinanceAssistantGraph, Depends(get_query_graph)],
) -> QueryResponse:
    from langgraph.types import Command

    result = await graph.ainvoke(
        Command(resume=request.reply),
        config={"configurable": {"thread_id": request.thread_id}},
    )
    interrupt_payload = await _get_interrupt_payload(graph, request.thread_id, result)
    status = "clarification" if interrupt_payload else result.get("status")
    if status in {None, "answered", "not_implemented"}:
        status = None
    return QueryResponse(
        intent=result.get("intent", "finance_qa"),
        answer=result.get("answer", ""),
        status=status,
        citations=result.get("citations") or None,
        thread_id=request.thread_id if interrupt_payload else None,
        handoff_intents=result.get("handoff_intents") or None,
        clarification_question=(
            interrupt_payload.get("question")
            if interrupt_payload
            else result.get("clarification_question") or None
        ),
    )


async def _get_interrupt_payload(
    graph: FinanceAssistantGraph,
    thread_id: str,
    result: dict[str, object],
) -> dict[str, object] | None:
    for item in result.get("__interrupt__", ()):
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("kind") == "clarification":
            return value
    if getattr(graph, "checkpointer", None) is None or getattr(graph, "checkpointer", None) is False:
        return None
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except ValueError:
        return None
    for task in snapshot.tasks:
        for item in task.interrupts:
            value = item.value
            if isinstance(value, dict) and value.get("kind") == "clarification":
                return value
    return None
