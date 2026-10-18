import json

import httpx
import pytest
from pydantic import ValidationError

from finassist.web_app.config import WebSettings
from finassist.web_app.services.api_client import ApiError, FinanceApiClient
from finassist.web_app.services.models import QueryRequest, QueryResponse, ResumeRequest


def test_requests_responses_and_timeouts() -> None:
    seen = []
    payload = {
        "intent": "goals",
        "answer": "More details?",
        "classified_intent": "goals",
        "executed_handler": "goals_node",
        "calculation": {"amount": 100, "nested": [True, None]},
        "calculations": [{"total": 20}],
        "status": "clarification",
        "citations": ["source"],
        "thread_id": "thread-1",
        "clarification_question": "When?",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"status": "ok", "version": "0.1.0"} if request.method == "GET" else payload
        )

    with FinanceApiClient(
        WebSettings("http://api.test/prefix/", read_timeout=90),
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.health().status == "ok"
        result = client.query(QueryRequest(query="  Save money  "))
        assert result.model_dump() == payload
        client.query(
            QueryRequest(
                query="Analyze", user_role="portfolio_manager", customer_id="C1", manager_id="M1"
            )
        )
        assert (
            client.resume(ResumeRequest(thread_id="thread-1", reply="  In two years  ")) == result
        )
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", "/prefix/health"),
        ("POST", "/prefix/query"),
        ("POST", "/prefix/query"),
        ("POST", "/prefix/query/resume"),
    ]
    assert json.loads(seen[1].content) == {"query": "Save money"}
    assert json.loads(seen[2].content) == {
        "query": "Analyze",
        "user_role": "portfolio_manager",
        "customer_id": "C1",
        "manager_id": "M1",
    }
    assert json.loads(seen[3].content) == {"thread_id": "thread-1", "reply": "In two years"}
    assert seen[0].extensions["timeout"]["read"] == 5
    assert seen[1].extensions["timeout"] == {"connect": 5, "read": 90, "write": 10, "pool": 5}


@pytest.mark.parametrize(
    "payload",
    [
        {"intent": "finance_qa", "answer": "An answer"},
        {"intent": "tax", "answer": "", "status": "error", "citations": None},
    ],
)
def test_optional_fields_and_application_status(payload: dict) -> None:
    with FinanceApiClient(
        WebSettings(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        result = client.query(QueryRequest(query="Explain"))
        assert result.answer == payload["answer"]
        assert result.status == payload.get("status")
        assert result.thread_id is None


@pytest.mark.parametrize("method", ["health", "query", "resume"])
@pytest.mark.parametrize(
    "failure,code",
    [
        ("offline", "offline"),
        ("timeout", "timeout"),
        (422, "http_error"),
        (403, "http_error"),
        (429, "http_error"),
        (500, "http_error"),
        (502, "http_error"),
        (503, "http_error"),
        (504, "http_error"),
        (302, "http_error"),
        ("bad_json", "invalid_response"),
        ("schema", "invalid_response"),
    ],
)
def test_safe_failures_without_retry(method: str, failure: str | int, code: str) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failure == "offline":
            raise httpx.ConnectError("SECRET connection details", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("SECRET question", request=request)
        if failure == "bad_json":
            return httpx.Response(200, text="SECRET HTML")
        if failure == "schema":
            return httpx.Response(200, json={"SECRET": "invalid"})
        return httpx.Response(
            int(failure),
            json={"detail": "SECRET server error"},
            headers={"location": "http://other.test"},
        )

    with FinanceApiClient(WebSettings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ApiError) as error:
            if method == "health":
                client.health()
            elif method == "query":
                client.query(QueryRequest(query="SECRET question"))
            else:
                client.resume(ResumeRequest(thread_id="SECRET thread", reply="SECRET reply"))
    assert len(calls) == 1
    assert error.value.code == code
    assert "SECRET" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.status_code == (failure if isinstance(failure, int) else None)


@pytest.mark.parametrize(
    "factory,payload",
    [
        (QueryRequest, {}),
        (QueryRequest, {"query": "  "}),
        (QueryRequest, {"query": "x" * 4001}),
        (QueryRequest, {"query": "x", "user_role": "admin"}),
        (QueryRequest, {"query": "x", "customer_id": ""}),
        (QueryRequest, {"query": "x", "manager_id": "x" * 101}),
        (ResumeRequest, {"thread_id": "", "reply": "yes"}),
        (ResumeRequest, {"thread_id": "t", "reply": " "}),
        (QueryResponse, {"intent": "invented", "answer": "x"}),
        (QueryResponse, {"intent": "tax", "answer": 42}),
        (QueryResponse, {"intent": "tax", "answer": "x", "citations": [1]}),
    ],
)
def test_wire_validation(factory: type, payload: dict) -> None:
    with pytest.raises(ValidationError):
        factory(**payload)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_timeout_environment(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("FINPILOT_READ_TIMEOUT", value)
    with pytest.raises(ValueError, match="Check FINPILOT"):
        WebSettings.from_environment()


def test_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINPILOT_API_BASE_URL", "https://api.test/v1/")
    monkeypatch.setenv("FINPILOT_READ_TIMEOUT", "45")
    settings = WebSettings.from_environment()
    assert settings.api_base_url == "https://api.test/v1"
    assert settings.read_timeout == 45


def test_models_match_committed_openapi_contract() -> None:
    from pathlib import Path

    from finassist.web_app.services.models import HealthResponse

    contract = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/api/finassist-openapi.json").read_text()
    )
    assert set(contract["paths"]) == {"/health", "/query", "/query/resume"}
    for model in (HealthResponse, QueryRequest, ResumeRequest, QueryResponse):
        actual = model.model_json_schema()
        expected = contract["components"]["schemas"][model.__name__]
        assert set(actual["properties"]) == set(expected["properties"])
        assert set(actual["required"]) == set(expected["required"])
        for name, field in expected["properties"].items():
            candidate = actual["properties"][name]
            # Ignore documentation and optional defaults; compare wire types and constraints.
            for key in ("type", "enum", "const", "minLength", "maxLength"):
                assert candidate.get(key) == field.get(key), (model.__name__, name, key)
            if "anyOf" in field:
                assert len(candidate["anyOf"]) == len(field["anyOf"])
                for left, right in zip(candidate["anyOf"], field["anyOf"], strict=True):
                    for key in ("type", "enum", "minLength", "maxLength"):
                        assert left.get(key) == right.get(key), (model.__name__, name, key)
