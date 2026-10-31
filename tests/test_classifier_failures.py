from types import SimpleNamespace
from unittest.mock import AsyncMock

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

from finassist.api.query import get_query_graph, get_simulated_user_context
from finassist.evaluation.router.evaluation import (
    RouterEvalCase,
    evaluate_classifier,
    evaluate_routes,
)
from finassist.main import create_app
from finassist.workflow.builder import build_graph
from finassist.workflow.classifier import (
    ClassificationOutputError,
    ClassificationProviderError,
    ClassificationTimeoutError,
    LLMIntentClassifier,
)


class Model:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.result


INVALID_RESULTS = [
    None,
    {"intent": "investment"},
    {"answer": "Private text"},
    {"intent": None},
    {"intent": ["tax"]},
    [],
    "not JSON",
    42,
]


@pytest.mark.parametrize("result", INVALID_RESULTS)
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


PROVIDER_FAILURES = [
    (APIConnectionError(request=httpx.Request("POST", "https://provider.example")), 503),
    (
        RateLimitError(
            "Private provider details",
            response=httpx.Response(429, request=httpx.Request("POST", "https://provider.example")),
            body=None,
        ),
        503,
    ),
    (OutputParserException("Private model output"), 502),
    (APITimeoutError(request=httpx.Request("POST", "https://provider.example")), 504),
    (
        AuthenticationError(
            "Private credentials",
            response=httpx.Response(401, request=httpx.Request("POST", "https://provider.example")),
            body=None,
        ),
        503,
    ),
    (
        InternalServerError(
            "Private upstream error",
            response=httpx.Response(500, request=httpx.Request("POST", "https://provider.example")),
            body=None,
        ),
        503,
    ),
]


@pytest.mark.parametrize("error,code", PROVIDER_FAILURES)
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


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/query", {"query": "Explain ETFs"}),
        ("/query/resume", {"thread_id": "paused-thread", "reply": "Five years"}),
    ],
)
@pytest.mark.parametrize(
    "error,code,detail",
    [
        (ClassificationTimeoutError("Private details"), 504,
         "Intent classification timed out. Please try again."),
        (ClassificationProviderError("Private details"), 503,
         "Intent classification provider unavailable. Please try again."),
        (ClassificationOutputError("Private details"), 502,
         "Intent classification returned an invalid response. Please try again."),
    ],
)
def test_classification_http_error_contract_on_query_and_resume(path, payload, error, code, detail):
    graph = SimpleNamespace(
        ainvoke=AsyncMock(side_effect=error),
        aget_state=AsyncMock(return_value=SimpleNamespace(
            values={"portfolio_user_context": None},
        )),
    )
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph
    app.dependency_overrides[get_simulated_user_context] = lambda: None

    response = TestClient(app).post(path, json=payload)

    assert response.status_code == code
    assert response.json() == {"detail": detail}
    assert "Private" not in response.text


async def test_classifier_failure_never_invokes_domain_agent():
    from types import SimpleNamespace

    from langchain_core.runnables import RunnableLambda

    from finassist.workflow.classifier import ClassificationOutputError

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


@pytest.mark.parametrize("evaluate", [evaluate_classifier, evaluate_routes])
@pytest.mark.parametrize("result", INVALID_RESULTS)
async def test_invalid_provider_output_is_recorded_by_evaluation(evaluate, result):
    await assert_evaluation_failure(evaluate, Model(result), "invalid_output")


@pytest.mark.parametrize("evaluate", [evaluate_classifier, evaluate_routes])
@pytest.mark.parametrize("error,code", PROVIDER_FAILURES)
async def test_provider_failure_is_recorded_by_evaluation(evaluate, error, code):
    category = {502: "invalid_output", 503: "provider_unavailable", 504: "timeout"}[code]
    await assert_evaluation_failure(evaluate, Model(error=error), category)


async def assert_evaluation_failure(evaluate, model, category):
    class RecoveringModel(Model):
        async def ainvoke(self, messages):
            if self.calls == 0:
                self.calls += 1
                return await model.ainvoke(messages)
            self.calls += 1
            return {"intent": "finance_qa"}

    recovering = RecoveringModel()
    recovering.calls = 0
    cases = [
        RouterEvalCase(
            id=str(index),
            query="Explain ETFs",
            expected_classified_intent="finance_qa",
            expected_handler="finance_qa_node",
            tags=["failure"],
            critical=True,
        )
        for index in range(2)
    ]
    results = await evaluate(cases, LLMIntentClassifier(recovering, "Classify"))
    failed, recovered = results
    assert failed.status == "error"
    assert failed.error_category == category
    assert failed.predicted_intent is None
    assert failed.predicted_handler is None
    assert failed.intent_matches is None
    assert failed.handler_matches is None
    assert failed.elapsed_ms >= 0
    assert "Private" not in failed.model_dump_json()
    assert recovered.status == "passed"
    assert recovering.calls == 2
