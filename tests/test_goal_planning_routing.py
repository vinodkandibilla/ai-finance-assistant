import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.graph.builder import build_graph
from ai_finance_assistant.main import create_app
from test_finance_autonomous_agent import call
from test_goal_planning_agent import make_agent


class GoalClassifier:
    def __init__(self):
        self.calls = []

    async def classify(self, query):
        self.calls.append(query)
        return "goals"


@pytest.mark.parametrize(
    "query,name,args,field,expected",
    [
        (
            "I want $100,000 in 5 years. How much should I invest monthly?",
            "calculate_required_contribution",
            {"target_amount": 100000, "time_horizon_years": 5},
            "required_monthly_contribution",
            1666.67,
        ),
        (
            "If I invest $500 monthly for 10 years at 7%, how much will I have?",
            "calculate_future_value",
            {"monthly_contribution": 500, "time_horizon_years": 10, "expected_annual_return": 7},
            "future_value",
            86542.4,
        ),
        (
            "If I invest $1,000 monthly, how long until I reach $100,000?",
            "calculate_time_to_goal",
            {"target_amount": 100000, "monthly_contribution": 1000},
            "months_to_goal",
            100,
        ),
        (
            "Can I reach $100,000 in 5 years investing $1,000 monthly?",
            "check_goal_feasibility",
            {"target_amount": 100000, "monthly_contribution": 1000, "time_horizon_years": 5},
            "feasible",
            False,
        ),
    ],
)
def test_goal_requests_reach_agent_and_return_calculation(query, name, args, field, expected):
    classifier = GoalClassifier()
    # Script the model decision; verify routing and actual tool execution without live API calls.
    agent, model = make_agent([call(name, args, "calc"), AIMessage(content="Done")])
    graph = build_graph(classifier, goal_agent=agent)
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph
    with TestClient(app) as client:
        response = client.post("/query", json={"query": query})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == body["classified_intent"] == "goals"
    assert body["executed_handler"] == "goals_node"
    assert body["calculation"][field] == expected
    assert classifier.calls == [query]
    executed = [m.name for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert executed == [name]


def test_goal_clarification_uses_existing_api_and_resumes_without_rerouting():
    classifier = GoalClassifier()
    agent, model = make_agent(
        [
            call(
                "ask_goal_clarification",
                {"question": "When would you like to reach your goal?"},
                "ask",
            ),
            call(
                "calculate_required_contribution",
                {"target_amount": 50000, "time_horizon_years": 3},
                "calc",
            ),
            AIMessage(content="Done"),
        ]
    )
    graph = build_graph(classifier, goal_agent=agent, checkpointer=MemorySaver())
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph
    query = "I want to save $50,000 for my daughter's college."
    with TestClient(app) as client:
        paused = client.post("/query", json={"query": query}).json()
        assert paused["status"] == "clarification"
        assert paused["executed_handler"] == "goals_node"
        completed = client.post(
            "/query/resume",
            json={
                "thread_id": paused["thread_id"],
                "reply": "In 3 years.",
            },
        ).json()
    assert completed["calculation"]["required_monthly_contribution"] == 1388.89
    assert completed["intent"] == "goals"
    assert "thread_id" not in completed
    assert classifier.calls == [query]
    assert any(m.content == query for m in model.seen[-1])
