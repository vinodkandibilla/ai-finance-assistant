"""Manager switching is request-local and limited to the simulated local workspace."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from finassist.api.query import get_query_graph
from finassist.core.config import get_settings
from finassist.main import create_app
from finassist.workflow.builder import build_graph


@pytest.fixture
def local_client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("PORTFOLIO_SIMULATED_USER_ID", "PM001")
    monkeypatch.setenv("PORTFOLIO_SIMULATED_USER_ROLE", "PORTFOLIO_MANAGER")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def test_manager_directory_and_invalid_selection(local_client):
    result = local_client.get("/portfolio/managers").json()
    assert len(result["managers"]) == 5
    assert result["current_manager_id"] == "PM001"
    assert result["can_switch"] is True
    assert set(result["managers"][0]) == {"manager_id", "manager_name"}
    assert (
        local_client.get(
            "/portfolio/managers", headers={"X-Portfolio-Manager": "PM999"}
        ).status_code
        == 422
    )


def test_selected_manager_is_request_local(local_client):
    async def classify(query):
        return "portfolio"

    async def portfolio(state, config):
        identity = config["configurable"]["portfolio_user_context"]
        return {"answer": identity["user_id"], "status": "answered"}

    local_client.app.dependency_overrides[get_query_graph] = lambda: build_graph(
        SimpleNamespace(classify=classify),
        portfolio_agent=SimpleNamespace(graph=RunnableLambda(portfolio)),
    )
    selected = local_client.post(
        "/query", json={"query": "Show my customers"}, headers={"X-Portfolio-Manager": "PM002"}
    )
    assert selected.json()["answer"] == "PM002"
    default = local_client.post("/query", json={"query": "Show my customers"})
    assert default.json()["answer"] == "PM001"
    assert local_client.app.state.portfolio_user_context.user_id == "PM001"


def test_switching_is_rejected_outside_local_mode(local_client, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    response = local_client.get("/portfolio/managers", headers={"X-Portfolio-Manager": "PM002"})
    assert response.status_code == 403
    result = local_client.get("/portfolio/managers").json()
    assert result["can_switch"] is False
    assert [m["manager_id"] for m in result["managers"]] == ["PM001"]


def test_customer_identity_cannot_switch_managers(local_client):
    from finassist.agents.portfolio_analysis.authorization import UserContext

    local_client.app.state.portfolio_user_context = UserContext(user_id="C001", role="CUSTOMER")
    assert local_client.get("/portfolio/managers").json()["managers"] == []
    assert (
        local_client.get(
            "/portfolio/managers", headers={"X-Portfolio-Manager": "PM002"}
        ).status_code
        == 403
    )


def test_resume_remains_bound_to_selected_manager(local_client):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt

    async def classify(query):
        return "portfolio"

    def portfolio(state):
        reply = interrupt({"kind": "clarification", "question": "Which analysis?"})
        return {"answer": str(reply), "status": "answered"}

    graph = build_graph(
        SimpleNamespace(classify=classify),
        portfolio_agent=SimpleNamespace(graph=RunnableLambda(portfolio)),
        checkpointer=MemorySaver(),
    )
    local_client.app.dependency_overrides[get_query_graph] = lambda: graph
    paused = local_client.post(
        "/query", json={"query": "Analyze my customers"}, headers={"X-Portfolio-Manager": "PM002"}
    ).json()
    request = {"thread_id": paused["thread_id"], "reply": "Value"}
    assert (
        local_client.post(
            "/query/resume", json=request, headers={"X-Portfolio-Manager": "PM003"}
        ).status_code
        == 403
    )
    assert (
        local_client.post(
            "/query/resume", json=request, headers={"X-Portfolio-Manager": "PM002"}
        ).json()["answer"]
        == "Value"
    )
