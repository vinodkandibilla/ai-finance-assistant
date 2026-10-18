from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from finassist.api.query import get_query_graph
from finassist.main import create_app
from finassist.web_app.services.showcase import scope_market_question
from finassist.workflow.builder import build_graph
from finassist.workflow.greetings import WELCOME, is_standalone_greeting


@pytest.mark.parametrize(
    "query",
    ["Hi", "Hello!", "Good morning", "Hi, FinPilot!", "Hey there", scope_market_question("Hello!")],
)
async def test_greetings_skip_classifier_and_domain_tools(query):
    classifier = AsyncMock()
    graph = build_graph(classifier)
    result = await graph.ainvoke({"query": query})
    assert result["answer"] == WELCOME
    assert not result.get("executed_handler")
    classifier.classify.assert_not_awaited()


@pytest.mark.parametrize(
    "query",
    [
        "Hi, what is an ETF?",
        "Hello, quote AAPL",
        "Good morning, show AAPL history",
        "highest stock price",
    ],
)
def test_greeting_with_question_is_not_intercepted(query):
    assert not is_standalone_greeting(query)


def test_greeting_response_uses_existing_api_schema():
    classifier = AsyncMock()
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(classifier)
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "Hello!"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == WELCOME
    assert "executed_handler" not in body
    assert "thread_id" not in body
    classifier.classify.assert_not_awaited()
