import logging
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langsmith import trace, traceable
from pydantic import BaseModel, Field, field_validator

from finassist.agents.common.answer_cache import SemanticAnswerCache, cache_namespace
from finassist.agents.finance_qa import FinanceQaAgent
from finassist.agents.finance_qa.answer_cache import definition_scope as finance_cache_scope
from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.corpus_index import get_finance_corpus
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.reranker import FinanceCrossEncoderReranker
from finassist.agents.finance_qa.retrieval import FinanceHybridRetriever
from finassist.agents.goal_planning import GoalPlanningAgent
from finassist.agents.market_analysis.agent import MarketAnalysisAgent
from finassist.agents.market_analysis.session import MarketSessionManager
from finassist.agents.news_synthesizer import NewsSynthesizerAgent
from finassist.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from finassist.agents.portfolio_analysis.authorization import UserContext
from finassist.agents.tax_education.answer_cache import definition_scope as tax_cache_scope
from finassist.agents.tax_education.config import TaxAgentSettings
from finassist.agents.tax_education.graph import TaxEducationAgent
from finassist.core.config import get_settings
from finassist.data.portfolio_store import PortfolioStore
from finassist.guardrails.pii import redact_pii
from finassist.prompts import FINANCE_QA_AGENT_PROMPT, TAX_EDUCATION_AGENT_PROMPT
from finassist.rag.qdrant import get_qdrant_client
from finassist.rag.tax_education import get_tax_retriever, load_tax_documents
from finassist.workflow.builder import FinanceAssistantGraph, build_graph
from finassist.workflow.classifier import (
    ClassificationOutputError,
    ClassificationProviderError,
    ClassificationTimeoutError,
    IntentClassifier,
    create_intent_classifier,
)
from finassist.workflow.state import ExecutedHandler, FinanceIntent

router = APIRouter(tags=["assistant"])
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    market_scope: Literal["us_nasdaq_usd"] | None = None
    user_role: Literal["portfolio_manager", "individual_customer"] | None = None
    customer_id: str | None = Field(default=None, min_length=1, max_length=100)
    manager_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class QueryResponse(BaseModel):
    intent: FinanceIntent
    classified_intent: FinanceIntent | None = None
    executed_handler: ExecutedHandler | None = None
    answer: str
    calculation: dict | None = None
    calculations: list[dict] | None = None
    status: str | None = None
    citations: list[str] | None = None
    thread_id: str | None = None
    clarification_question: str | None = None


async def get_query_checkpointer(request: Request) -> AsyncSqliteSaver:
    return request.app.state.query_checkpointer


async def get_portfolio_store(request: Request) -> PortfolioStore:
    return request.app.state.portfolio_store


async def get_simulated_user_context(
    request: Request,
    selected: Annotated[
        str | None,
        Header(
            alias="X-Portfolio-Manager",
            description="Local simulated manager selection; unavailable outside local mode.",
        ),
    ] = None,
) -> UserContext | None:
    identity = getattr(request.app.state, "portfolio_user_context", None)
    if selected:
        if (
            get_settings().app_env != "local"
            or identity is None
            or identity.role != "PORTFOLIO_MANAGER"
        ):
            raise HTTPException(status_code=403, detail="Manager selection is unavailable.")
        managers = request.app.state.portfolio_store.get_managers()
        if selected not in {row["manager_id"] for row in managers}:
            raise HTTPException(status_code=422, detail="Unknown portfolio manager.")
        return UserContext(user_id=selected, role="PORTFOLIO_MANAGER")
    return identity


class PortfolioManager(BaseModel):
    manager_id: str
    manager_name: str


class PortfolioManagerDirectory(BaseModel):
    managers: list[PortfolioManager]
    current_manager_id: str | None = None
    can_switch: bool = False


@router.get("/portfolio/managers", response_model=PortfolioManagerDirectory)
async def portfolio_managers(
    store: Annotated[PortfolioStore, Depends(get_portfolio_store)],
    identity: Annotated[UserContext | None, Depends(get_simulated_user_context)],
) -> PortfolioManagerDirectory:
    if identity is None or identity.role != "PORTFOLIO_MANAGER":
        return PortfolioManagerDirectory(managers=[])
    can_switch = get_settings().app_env == "local"
    managers = [
        PortfolioManager.model_validate(row)
        for row in store.get_managers()
        if can_switch or row["manager_id"] == identity.user_id
    ]
    return PortfolioManagerDirectory(
        managers=managers, current_manager_id=identity.user_id, can_switch=can_switch
    )


def get_market_sessions(request: Request) -> MarketSessionManager | None:
    return getattr(request.app.state, "market_sessions", None)


@lru_cache
def get_query_graph(
    checkpointer: Annotated[AsyncSqliteSaver | None, Depends(get_query_checkpointer)] = None,
    portfolio_store: Annotated[PortfolioStore | None, Depends(get_portfolio_store)] = None,
    market_sessions: Annotated[MarketSessionManager | None, Depends(get_market_sessions)] = None,
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
    finance_answer_cache = None
    tax_answer_cache = None
    tax_agent = None
    llm = None
    if settings.openai_api_key is not None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        qdrant_client = get_qdrant_client()
        tax_retriever = get_tax_retriever(qdrant_client, llm)

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
            reranker=FinanceCrossEncoderReranker(finance_settings.reranker_model)
            if finance_settings.reranker_enabled
            else None,
            settings=finance_settings,
        )
        if finance_settings.answer_cache_enabled:
            finance_answer_cache = SemanticAnswerCache(
                OpenAIEmbeddings(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key.get_secret_value(),
                    max_retries=2,
                    timeout=finance_settings.answer_cache_embedding_timeout_seconds,
                ),
                scope=finance_cache_scope,
                client=qdrant_client,
                collection=finance_settings.answer_cache_collection,
                namespace=cache_namespace(
                    schema=2,
                    prompt=FINANCE_QA_AGENT_PROMPT,
                    model=settings.finance_query_analysis_model,
                    embedding_model=settings.openai_embedding_model,
                    settings=finance_settings.model_dump(),
                    corpus=[
                        {"content": d.page_content, "metadata": d.metadata} for d in finance_corpus
                    ],
                ),
                allowed_citations={
                    f"Financial Education Handbook, page {d.metadata['page_number']}"
                    for d in finance_corpus
                },
                ttl_seconds=finance_settings.answer_cache_ttl_seconds,
                similarity=finance_settings.answer_cache_similarity,
                embedding_timeout_seconds=finance_settings.answer_cache_embedding_timeout_seconds,
            )
        tax_settings = TaxAgentSettings()
        tax_agent = TaxEducationAgent(
            retriever=tax_retriever,
            llm=llm,
            settings=tax_settings,
            reranker=FinanceCrossEncoderReranker(settings.finance_reranker_model)
            if tax_settings.reranker_enabled
            else None,
        )
        if tax_settings.answer_cache_enabled:
            corpus = load_tax_documents()
            tax_answer_cache = SemanticAnswerCache(
                OpenAIEmbeddings(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key.get_secret_value(),
                    max_retries=2,
                    timeout=tax_settings.answer_cache_embedding_timeout_seconds,
                ),
                client=qdrant_client,
                scope=tax_cache_scope,
                intent="tax",
                label="Tax",
                collection=tax_settings.answer_cache_collection,
                namespace=cache_namespace(
                    schema=1,
                    intent="tax",
                    jurisdiction="US_federal",
                    prompt=TAX_EDUCATION_AGENT_PROMPT,
                    model=settings.openai_model,
                    embedding_model=settings.openai_embedding_model,
                    settings=tax_settings.model_dump(),
                    corpus=[{"content": d.page_content, "metadata": d.metadata} for d in corpus],
                ),
                allowed_citations={
                    str(d.metadata.get("title") or "Tax education corpus") for d in corpus
                },
                ttl_seconds=tax_settings.answer_cache_ttl_seconds,
                similarity=tax_settings.answer_cache_similarity,
                embedding_timeout_seconds=tax_settings.answer_cache_embedding_timeout_seconds,
            )
    return build_graph(
        classifier,
        market_agent=MarketAnalysisAgent(llm=llm, settings=settings, sessions=market_sessions),
        finance_agent=finance_agent,
        finance_answer_cache=finance_answer_cache,
        goal_agent=GoalPlanningAgent(llm=llm),
        news_agent=NewsSynthesizerAgent(llm=llm, settings=settings),
        portfolio_agent=PortfolioAnalysisAgent(llm=llm, store=portfolio_store)
        if portfolio_store is not None
        else None,
        tax_agent=tax_agent,
        tax_answer_cache=tax_answer_cache,
        checkpointer=checkpointer,
    )


def _trace_config(source: str, *, pii_entities: tuple[str, ...] = ()) -> dict:
    settings = get_settings()
    if not settings.trace_tags_enabled:
        return {}
    return {
        "tags": [
            settings.trace_app_tag,
            f"env:{settings.trace_environment or settings.app_env}",
            f"source:{'web-chat' if source == 'web-chat' else 'api'}",
        ],
        "metadata": {
            "finpilot_agent_tags": True,
            "prompt_version": settings.intent_router_prompt_version,
            "pii_redacted": bool(pii_entities),
            "pii_entity_types": list(pii_entities),
        },
    }


def _trace_regex_block(*, source: str, reason: str, query_length: int) -> None:
    """Create a redacted guardrail trace without recording query contents."""
    settings = get_settings()
    if not settings.trace_tags_enabled:
        return
    with trace(
        "regex_guard",
        run_type="chain",
        inputs={"query_length": query_length},
        tags=[
            settings.trace_app_tag,
            f"env:{settings.trace_environment or settings.app_env}",
            f"source:{'web-chat' if source == 'web-chat' else 'api'}",
            "guardrail",
        ],
        metadata={
            "guardrail_blocked": True,
            "guardrail_reason": reason,
            "prompt_version": settings.intent_router_prompt_version,
        },
    ) as span:
        span.add_outputs({"status": "blocked", "reason": reason})


def _trace_guardrail(name: str, *, decision: str, metadata: dict | None = None) -> None:
    settings = get_settings()
    if not settings.trace_tags_enabled:
        return
    with trace(
        name,
        run_type="chain",
        inputs={},
        tags=[settings.trace_app_tag, "guardrail"],
        metadata={"decision": decision, **(metadata or {})},
    ) as span:
        span.add_outputs({"decision": decision})


def _redact_output(text: str) -> str:
    """Sanitize the final user-visible answer using the input PII guard."""
    redacted = redact_pii(text)
    _trace_guardrail(
        "output_guard",
        decision="redacted" if redacted.entities else "allowed",
        metadata={"entity_types": list(redacted.entities)},
    )
    return redacted.text


def _finalize_response(response: QueryResponse, raw_status: str | None) -> QueryResponse:
    """Apply the small set of shared MVP response checks."""
    if (
        response.status == "clarification"
        and response.thread_id
        and response.clarification_question
    ):
        # Nested graph interrupts can precede any parent answer/status update.
        # A valid paused question must not become an empty-answer fallback.
        if not response.answer.strip():
            response = response.model_copy(update={
                "answer": _redact_output(response.clarification_question)
            })
        return QueryResponse.model_validate(response.model_dump(exclude_none=True))
    if response.clarification_question and not response.thread_id:
        response = response.model_copy(update={"clarification_question": None})
        if response.status == "clarification":
            response = response.model_copy(update={"status": "insufficient"})
    if response.status == "insufficient" and not response.answer.strip():
        response = response.model_copy(update={
            "answer": "I could not find sufficient evidence to answer all parts of your question. "
            "Please narrow the question or try again."
        })
    if raw_status in {None, "answered", "not_implemented"} and not response.answer.strip():
        _trace_guardrail("output_guard", decision="fallback", metadata={"reason": "empty_answer"})
        return QueryResponse(
            intent=response.intent,
            classified_intent=response.classified_intent,
            executed_handler=response.executed_handler,
            answer="I could not produce a verified answer from the available information.",
            status="insufficient",
            thread_id=response.thread_id,
        )
    # Re-validate the public boundary model after all sanitization.
    return QueryResponse.model_validate(response.model_dump(exclude_none=True))


def _safe_error_response(error: Exception) -> QueryResponse:
    """Log details internally and expose only a safe MVP error message."""
    logger.exception("assistant_execution_failed", exc_info=error)
    _trace_guardrail("output_guard", decision="fallback", metadata={"reason": "execution_error"})
    if isinstance(error, ClassificationTimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Intent classification timed out. Please try again.",
        ) from None
    if isinstance(error, ClassificationProviderError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Intent classification provider unavailable. Please try again.",
        ) from None
    if isinstance(error, ClassificationOutputError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Intent classification returned an invalid response. Please try again.",
        ) from None
    error_name = type(error).__name__.lower()
    detail = (
        "The service is temporarily unavailable. Please try again shortly."
        if "ratelimit" in error_name or "quota" in error_name
        else "The request could not be completed right now."
    )
    return QueryResponse(intent="other", answer=detail, status="error")


def _trace_usage_summary(result: dict) -> None:
    """Aggregate token metadata from returned model messages without logging content."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "model_calls": 0}
    for message in result.get("messages", []):
        usage = (
            message.get("usage_metadata", {})
            if isinstance(message, dict)
            else getattr(message, "usage_metadata", None)
        ) or {}
        if usage:
            totals["input_tokens"] += int(usage.get("input_tokens", 0))
            totals["output_tokens"] += int(usage.get("output_tokens", 0))
            totals["total_tokens"] += int(usage.get("total_tokens", 0))
            totals["model_calls"] += 1
    if not totals["model_calls"] or not get_settings().trace_tags_enabled:
        return
    with trace(
        "request_cost_summary",
        run_type="chain",
        inputs={},
        metadata={"token_usage_summary": True},
    ) as span:
        span.add_outputs(totals)


def _trace_request_inputs(inputs: dict) -> dict:
    """Expose only redacted user text, never dependencies or credentials."""
    request = inputs.get("request")
    if isinstance(request, QueryRequest):
        return {"query": redact_pii(request.query).text}
    if isinstance(request, ResumeRequest):
        return {"reply": redact_pii(request.reply).text}
    return {}


def _trace_response_outputs(output: object) -> dict:
    """Preview the public answer without capturing structured account payloads."""
    if not isinstance(output, QueryResponse):
        return {}
    return {
        "answer": redact_pii(output.answer).text,
        "intent": output.intent,
        "status": output.status,
    }


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
@traceable(
    name="FinPilot request",
    run_type="chain",
    process_inputs=_trace_request_inputs,
    process_outputs=_trace_response_outputs,
)
async def query_assistant(
    request: QueryRequest,
    graph: Annotated[FinanceAssistantGraph, Depends(get_query_graph)],
    user_context: Annotated[UserContext | None, Depends(get_simulated_user_context)] = None,
    source: Annotated[str, Header(alias="X-FinPilot-Source")] = "api",
) -> QueryResponse:
    import uuid

    thread_id = str(uuid.uuid4())
    redacted = redact_pii(request.query)
    try:
        result = await graph.ainvoke(
            {
                "query": redacted.text,
                "thread_id": thread_id,
                "portfolio_user_context": user_context.model_dump() if user_context else None,
                **request.model_dump(exclude_none=True, exclude={"query"}),
            },
            config={
                **_trace_config(source, pii_entities=redacted.entities),
                "configurable": {
                    "thread_id": thread_id,
                    "portfolio_user_context": user_context.model_dump() if user_context else None,
                },
            },
        )
    except Exception as error:
        return _safe_error_response(error)
    _trace_usage_summary(result)
    interrupt_payload = await _get_interrupt_payload(graph, thread_id, result)
    status = "clarification" if interrupt_payload else result.get("status")
    if status in {None, "answered", "not_implemented"}:
        status = None
    answer = (
        (interrupt_payload.get("answer") or result.get("answer", ""))
        if interrupt_payload
        else result.get("answer", "")
    )
    response = QueryResponse(
        intent=result["intent"],
        classified_intent=result.get("classified_intent"),
        executed_handler=result.get("executed_handler"),
        answer=_redact_output(answer),
        calculations=result.get("calculations"),
        calculation=interrupt_payload.get("calculation")
        if interrupt_payload
        else result.get("calculation"),
        status=status,
        citations=(interrupt_payload.get("citations") or result.get("citations") or None)
        if interrupt_payload
        else result.get("citations") or None,
        thread_id=thread_id if interrupt_payload else None,
        clarification_question=(
            interrupt_payload.get("question")
            if interrupt_payload
            else result.get("clarification_question") or None
        ),
    )
    return _finalize_response(response, result.get("status"))


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    reply: str = Field(min_length=1, max_length=4000)

    @field_validator("reply", mode="before")
    @classmethod
    def strip_reply(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


@router.post("/query/resume", response_model=QueryResponse, response_model_exclude_none=True)
@traceable(
    name="FinPilot resume",
    run_type="chain",
    process_inputs=_trace_request_inputs,
    process_outputs=_trace_response_outputs,
)
async def resume_query(
    request: ResumeRequest,
    graph: Annotated[FinanceAssistantGraph, Depends(get_query_graph)],
    user_context: Annotated[UserContext | None, Depends(get_simulated_user_context)] = None,
    source: Annotated[str, Header(alias="X-FinPilot-Source")] = "api",
) -> QueryResponse:
    from langgraph.types import Command

    identity = user_context.model_dump() if user_context else None
    snapshot = await graph.aget_state({"configurable": {"thread_id": request.thread_id}})
    if snapshot.values.get("portfolio_user_context") != identity:
        raise HTTPException(status_code=403, detail="You do not have access to this query thread.")
    redacted = redact_pii(request.reply)
    try:
        result = await graph.ainvoke(
            Command(resume=redacted.text),
            config={
                **_trace_config(source, pii_entities=redacted.entities),
                "configurable": {
                    "thread_id": request.thread_id,
                    "portfolio_user_context": identity,
                    "bypass_answer_cache": True,
                },
            },
        )
    except Exception as error:
        return _safe_error_response(error)
    _trace_usage_summary(result)
    interrupt_payload = await _get_interrupt_payload(graph, request.thread_id, result)
    status = "clarification" if interrupt_payload else result.get("status")
    if status in {None, "answered", "not_implemented"}:
        status = None
    answer = (
        (interrupt_payload.get("answer") or result.get("answer", ""))
        if interrupt_payload
        else result.get("answer", "")
    )
    response = QueryResponse(
        intent=result.get("intent", "finance_qa"),
        classified_intent=result.get("classified_intent"),
        executed_handler=result.get("executed_handler"),
        answer=_redact_output(answer),
        calculations=result.get("calculations"),
        calculation=interrupt_payload.get("calculation")
        if interrupt_payload
        else result.get("calculation"),
        status=status,
        citations=(interrupt_payload.get("citations") or result.get("citations") or None)
        if interrupt_payload
        else result.get("citations") or None,
        thread_id=request.thread_id if interrupt_payload else None,
        clarification_question=(
            interrupt_payload.get("question")
            if interrupt_payload
            else result.get("clarification_question") or None
        ),
    )
    return _finalize_response(response, result.get("status"))


async def _get_interrupt_payload(
    graph: FinanceAssistantGraph,
    thread_id: str,
    result: dict[str, object],
) -> dict[str, object] | None:
    for item in result.get("__interrupt__", ()):
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("kind") == "clarification":
            return value
    if (
        getattr(graph, "checkpointer", None) is None
        or getattr(graph, "checkpointer", None) is False
    ):
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
