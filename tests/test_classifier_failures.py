import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.graph.builder import build_graph
from ai_finance_assistant.graph.classifier import LLMIntentClassifier
from ai_finance_assistant.main import create_app


class Model:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"intent": "investment"},
        {"answer": "Private text"},
        {"intent": None},
        {"intent": ["tax"]},
        [],
        "not JSON",
        42,
    ],
)
def test_invalid_result_returns_502_without_dispatch(result):
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(
        LLMIntentClassifier(Model(result), "Classify")
    )
    response = TestClient(app).post("/query", json={"query": "Explain ETFs"})
    assert response.status_code == 502
    assert response.json() == {
        "detail": "Intent classification returned an invalid response. Please try again."
    }


@pytest.mark.parametrize(
    "error,code",
    [
        (APIConnectionError(request=httpx.Request("POST", "https://provider.example")), 503),
        (
            RateLimitError(
                "Private provider details",
                response=httpx.Response(
                    429, request=httpx.Request("POST", "https://provider.example")
                ),
                body=None,
            ),
            503,
        ),
        (OutputParserException("Private model output"), 502),
        (APITimeoutError(request=httpx.Request("POST", "https://provider.example")), 504),
        (
            AuthenticationError(
                "Private credentials",
                response=httpx.Response(
                    401, request=httpx.Request("POST", "https://provider.example")
                ),
                body=None,
            ),
            503,
        ),
        (
            InternalServerError(
                "Private upstream error",
                response=httpx.Response(
                    500, request=httpx.Request("POST", "https://provider.example")
                ),
                body=None,
            ),
            503,
        ),
    ],
)
def test_provider_and_parser_failures_have_separate_safe_responses(error, code):
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(
        LLMIntentClassifier(Model(error=error), "Classify")
    )
    response = TestClient(app).post("/query", json={"query": "Explain ETFs"})
    assert response.status_code == code
    assert "Private" not in response.text
    assert set(response.json()) == {"detail"}
    assert "intent" not in response.json()
    assert "executed_handler" not in response.json()


async def test_valid_structured_intent_still_succeeds():
    assert (
        await LLMIntentClassifier(Model({"intent": "finance_qa"}), "Classify").classify(
            "Explain ETFs"
        )
        == "finance_qa"
    )


async def test_classifier_failure_never_invokes_domain_agent():
    from types import SimpleNamespace

    from langchain_core.runnables import RunnableLambda

    from ai_finance_assistant.graph.classifier import ClassificationOutputError

    calls = []

    def handler(state):
        calls.append(state)
        return {"answer": "Should not run"}

    graph = build_graph(
        LLMIntentClassifier(Model({"intent": "invalid"}), "Classify"),
        finance_agent=SimpleNamespace(graph=RunnableLambda(handler)),
        tax_agent=SimpleNamespace(graph=RunnableLambda(handler)),
    )
    with pytest.raises(ClassificationOutputError):
        await graph.ainvoke({"query": "Explain ETFs"})
    assert not calls
