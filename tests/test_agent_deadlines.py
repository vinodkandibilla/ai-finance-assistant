import asyncio
import logging
import time

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.graph import FinanceQaAgent
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.schemas import FinanceAgentResult
from finassist.agents.tax_education.config import TaxAgentSettings
from finassist.agents.tax_education.schemas import TaxAgentResult
from finassist.workflow.builder import build_graph
from test_finance_autonomous_agent import Classifier, Reranker, Retriever, ScriptedModel, call
from test_tax_agent_graph import Classifier as TaxClassifier
from test_tax_agent_graph import make_agent


def agent_for(domain, timeout):
    if domain == "tax":
        return make_agent(
            [AIMessage(content="Done")],
            TaxAgentResult(status="answered", answer="Done"),
            agent_timeout_seconds=timeout,
        )[0]
    return FinanceQaAgent(
        llm=ScriptedModel(
            responses=[AIMessage(content="Done")],
            final=FinanceAgentResult(status="answered", answer="Done"),
        ),
        retriever=Retriever([]),
        reranker=Reranker(),
        classifier=Classifier(),
        metadata_extractor=FinanceMetadataExtractor([]),
        settings=FinanceAgentSettings(agent_timeout_seconds=timeout),
    )


@pytest.mark.parametrize("domain", ["finance_qa", "tax"])
async def test_deadline_cancels_loop_and_returns_safe_error(domain, caplog):
    agent = agent_for(domain, 0.03)
    cancelled = asyncio.Event()

    async def slow(_):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    agent.agent = RunnableLambda(slow)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        result = await agent.ainvoke({"query": "private query value"}, {})
    assert time.monotonic() - started < 1
    assert cancelled.is_set()
    assert result["status"] == "error"
    assert result["errors"] == ["agent_timeout"]
    assert not result["citations"]
    assert "private query value" not in caplog.text
    record = next(r for r in caplog.records if "agent_execution_failed" in r.message)
    assert record.failure_category == "timeout"
    assert record.timeout_ms == 30
    assert record.elapsed_ms >= 30


async def test_one_budget_covers_loop_and_completion_repair():
    agent = agent_for("tax", 0.1)
    cancelled = asyncio.Event()

    async def loop(_):
        try:
            await asyncio.sleep(0.06)
            return {"messages": [AIMessage(content="Missing terminal submission")]}
        finally:
            cancelled.set()

    agent.agent = RunnableLambda(loop)
    assert not hasattr(agent, "_formatter")
    result = await agent.ainvoke({"query": "Tax question"}, {})
    assert result["errors"] == ["agent_timeout"]
    assert cancelled.is_set()


async def test_clarification_wait_is_excluded_and_resume_has_fresh_budget():
    agent, _, _ = make_agent(
        [
            call("ask_tax_clarification", {"question": "Which topic?"}, "clarify"),
            call("tax_rag_query", {"query": "deductions"}, "rag"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Supported"),
        agent_timeout_seconds=0.2,
    )
    graph = build_graph(TaxClassifier(), tax_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "deadline-resume"}}
    paused = await graph.ainvoke({"query": "Tax help"}, config)
    assert paused.get("__interrupt__")
    await asyncio.sleep(0.25)
    result = await graph.ainvoke(Command(resume="Deductions"), config)
    assert result["status"] == "answered"


@pytest.mark.parametrize("settings", [FinanceAgentSettings, TaxAgentSettings])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_deadlines_rejected(settings, value):
    with pytest.raises(ValidationError):
        settings(agent_timeout_seconds=value)


@pytest.mark.parametrize(
    "settings,prefix", [(FinanceAgentSettings, "FINANCE"), (TaxAgentSettings, "TAX")]
)
def test_deadline_environment_configuration(settings, prefix, monkeypatch):
    monkeypatch.setenv(f"{prefix}_AGENT_TIMEOUT_SECONDS", "12.5")
    assert settings().agent_timeout_seconds == 12.5


async def test_finance_deadline_cancels_actual_pending_retrieval(monkeypatch):
    from test_finance_autonomous_agent import make_agent as make_finance_agent

    agent, retriever, _ = make_finance_agent(
        [call("finance_rag_query", {"query": "ETF"}, "search"), AIMessage(content="Done")],
        FinanceAgentResult(status="answered", answer="Invented answer"),
        agent_timeout_seconds=0.3,
    )
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def slow_retrieve(*args, **kwargs):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    monkeypatch.setattr(retriever, "retrieve", slow_retrieve)
    result = await agent.graph.ainvoke({"query": "What is an ETF?"})
    assert started.is_set()
    assert cancelled.is_set()
    assert result["status"] == "error"
    assert result["errors"] == ["agent_timeout"]
    assert not result["citations"]
    assert "Invented answer" not in result["answer"]
