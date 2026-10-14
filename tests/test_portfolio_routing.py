"""Router contract and real portfolio clarification/API integration without live models."""

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from finassist.api.query import get_query_graph
from finassist.data.portfolio_store import PortfolioStore
from finassist.main import create_app
from finassist.prompts import INTENT_ROUTER_SYSTEM_PROMPT
from finassist.workflow.builder import build_graph
from finassist.workflow.classifier import LLMIntentClassifier
from test_finance_autonomous_agent import call
from test_portfolio_agent import make_agent


class ClassifierModel:
    def __init__(self, intent):
        self.intent = intent
        self.seen = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.seen.append(messages)
        return {"intent": self.intent}


@pytest.mark.parametrize(
    "query,intent,handler",
    [
        ("Analyze my portfolio.", "portfolio", "portfolio_analysis_node"),
        ("Show my portfolio allocation.", "portfolio", "portfolio_analysis_node"),
        ("How is my portfolio performing?", "portfolio", "portfolio_analysis_node"),
        (
            "Which of my customers have concentrated portfolios?",
            "portfolio",
            "portfolio_analysis_node",
        ),
        ("What percentage of my portfolio is Apple?", "portfolio", "portfolio_analysis_node"),
        ("What is Apple's current stock price?", "market", "market_analysis_node"),
        ("What is an ETF?", "finance_qa", "finance_qa_node"),
        ("What is the latest Apple news?", "news", "news_synthesizer_node"),
    ],
)
async def test_documented_examples_retain_domain_routes(query, intent, handler, monkeypatch):
    # Scripted classification checks plumbing and prompt examples, not live-model accuracy.
    monkeypatch.setenv("MARKET_QUOTES_ENABLED", "true")
    from finassist.core.config import get_settings

    get_settings.cache_clear()
    try:
        model = ClassifierModel(intent)
        classifier = LLMIntentClassifier(model, INTENT_ROUTER_SYSTEM_PROMPT)
        result = await build_graph(classifier).ainvoke({"query": query})
        assert result["classified_intent"] == intent
        assert result["executed_handler"] == handler
        assert query in model.seen[0][0].content
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "query,question,reply,tool,args",
    [
        (
            "Analyze my portfolio.",
            "Which customer ID?",
            "C001",
            "get_customer_portfolio",
            {"customer_id": "C001"},
        ),
        (
            "Which of my customers have concentrated portfolios?",
            "Which manager ID?",
            "PM001",
            "screen_manager_portfolios",
            {"manager_id": "PM001"},
        ),
    ],
)
async def test_api_missing_scope_clarifies_and_resumes_same_domain(
    query, question, reply, tool, args
):
    with PortfolioStore() as store:
        model = ClassifierModel("portfolio")
        classifier = LLMIntentClassifier(model, INTENT_ROUTER_SYSTEM_PROMPT)
        agent, _ = make_agent(
            store,
            [
                call("ask_portfolio_clarification", {"question": question}, "ask"),
                call(tool, args, "data"),
                AIMessage(content="Done"),
            ],
        )
        graph = build_graph(classifier, portfolio_agent=agent, checkpointer=MemorySaver())
        app = create_app()
        app.dependency_overrides[get_query_graph] = lambda: graph
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            paused = (await client.post("/query", json={"query": query})).json()
            assert paused["status"] == "clarification"
            assert paused["clarification_question"] == question
            assert paused["executed_handler"] == "portfolio_analysis_node"
            assert "calculation" not in paused
            completed = (
                await client.post(
                    "/query/resume",
                    json={
                        "thread_id": paused["thread_id"],
                        "reply": reply,
                    },
                )
            ).json()
        assert completed["executed_handler"] == "portfolio_analysis_node"
        assert completed["intent"] == "portfolio"
        assert completed["calculation"]["status"] == "success"
        assert len(model.seen) == 1
