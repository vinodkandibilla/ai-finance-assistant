import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from finassist.api import query as query_api
from finassist.api.query import get_query_graph
from finassist.core.config import Settings
from finassist.main import create_app
from finassist.workflow.builder import (
    CLARIFY_RESPONSE,
    MARKET_QUOTE_UNAVAILABLE_RESPONSE,
    PORTFOLIO_ANALYSIS_RESPONSE,
    build_graph,
)


class FakeClassifier:
    def __init__(self, intent: str = "other") -> None:
        self.intent = intent

    async def classify(self, query: str) -> str:
        return self.intent


def test_query_endpoint_rejects_blank_query_without_llm_call() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier())
    client = TestClient(app)

    response = client.post("/query", json={"query": "  "})

    assert response.status_code == 422


def test_query_endpoint_routes_financial_education_to_empty_node() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("finance_qa"))
    client = TestClient(app)

    response = client.post("/query", json={"query": "Explain how an ETF works."})

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "finance_qa",
        "executed_handler": "finance_qa_node",
        "intent": "finance_qa",
        "answer": "",
    }


def test_query_endpoint_routes_portfolio_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("portfolio"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "Analyze my portfolio allocation and concentration."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "portfolio",
        "executed_handler": "portfolio_analysis_node",
        "intent": "portfolio",
        "answer": PORTFOLIO_ANALYSIS_RESPONSE,
        "status": "error",
    }


def test_query_endpoint_routes_market_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("market"))
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is the current price of AAPL?"})

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "market",
        "executed_handler": "market_analysis_node",
        "intent": "market",
        "answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE,
        "status": "error",
    }


def test_query_graph_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        query_api,
        "get_settings",
        lambda: Settings(_env_file=None, openai_api_key=None),
    )
    get_query_graph.cache_clear()

    try:
        with pytest.raises(HTTPException) as error:
            get_query_graph()
        assert get_query_graph.cache_info().currsize == 0
    finally:
        get_query_graph.cache_clear()

    assert error.value.status_code == 503


def test_query_graph_does_not_cache_when_llm_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_api,
        "get_settings",
        lambda: Settings(_env_file=None, openai_api_key="test-key"),
    )
    monkeypatch.setattr(query_api, "create_intent_classifier", lambda settings: FakeClassifier())
    calls = 0

    def fail_to_create_llm(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("LLM initialization failed")

    monkeypatch.setattr(query_api, "ChatOpenAI", fail_to_create_llm)
    get_query_graph.cache_clear()

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="LLM initialization failed"):
                get_query_graph()
        assert calls == 2
        assert get_query_graph.cache_info().currsize == 0
    finally:
        get_query_graph.cache_clear()


def test_settings_read_router_prompt_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENT_ROUTER_SYSTEM_PROMPT", "Use the configured test prompt.")

    settings = Settings(_env_file=None)

    assert settings.intent_router_system_prompt == "Use the configured test prompt."


def test_query_endpoint_returns_unconfigured_goals_error() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("goals"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "How much should I save each month?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "goals",
        "executed_handler": "goals_node",
        "intent": "goals",
        "answer": "Goal planning agent is not configured.",
        "status": "error",
    }


def test_query_endpoint_routes_news_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("news"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "Give me a concise summary of today's top AI news."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "news",
        "executed_handler": "news_synthesizer_node",
        "intent": "news",
        "answer": "News synthesizer agent is not configured.",
        "status": "error",
    }


def test_query_endpoint_routes_tax_intent() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("tax"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "What is the standard deduction for Married Filing Jointly "
            "with two qualifying children for the year 2026?"
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "tax",
        "executed_handler": "tax_education_node",
        "intent": "tax",
        "answer": "Tax education agent is not configured.",
        "status": "error",
    }


def test_query_endpoint_routes_clarify_intent_for_incomplete_details() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("clarify"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "What is the stock price?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "clarify",
        "executed_handler": "clarify_node",
        "intent": "clarify",
        "answer": CLARIFY_RESPONSE,
    }


def test_query_endpoint_routes_generic_market_query_to_clarify() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("market"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "What is the stock price?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "market",
        "executed_handler": "clarify_node",
        "intent": "market",
        "answer": CLARIFY_RESPONSE,
    }


def test_query_endpoint_routes_generic_news_query_to_news_node() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("news"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "top news"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "news",
        "executed_handler": "news_synthesizer_node",
        "intent": "news",
        "answer": "News synthesizer agent is not configured.",
        "status": "error",
    }


def test_query_endpoint_routes_generic_tax_query_to_tax_node() -> None:
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("tax"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "tax details"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classified_intent": "tax",
        "executed_handler": "tax_education_node",
        "intent": "tax",
        "answer": "Tax education agent is not configured.",
        "status": "error",
    }


def test_out_of_scope_response_has_no_handoff_contract():
    class Graph:
        async def ainvoke(self, state, config):
            return {
                "intent": "tax",
                "status": "out_of_scope",
                "answer": "This request is outside my tax-education scope. "
                "Please submit it as a new question.",
            }

    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: Graph()
    response = TestClient(app).post("/query", json={"query": "New task"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "out_of_scope"
    assert body["intent"] == "tax"
    assert "handoff_intents" not in body
    assert "thread_id" not in body
    assert "citations" not in body
    assert (
        "handoff_intents"
        not in app.openapi()["components"]["schemas"]["QueryResponse"]["properties"]
    )


def test_router_does_not_override_validated_classifier_decision():
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(FakeClassifier("finance_qa"))
    response = TestClient(app).post("/query", json={"query": "Explain Roth IRA contribution rules"})
    assert response.status_code == 200
    body = response.json()
    assert body["classified_intent"] == "finance_qa"
    assert body["executed_handler"] == "finance_qa_node"
    assert body["answer"] == ""
