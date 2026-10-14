from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ai_finance_assistant.api.query import get_query_graph
from ai_finance_assistant.main import create_app
from ai_finance_assistant.workflow.builder import build_graph


class Classifier:
    def __init__(self):
        self.calls = 0

    async def classify(self, query):
        self.calls += 1
        return "market"


class MarketAgent:
    def __init__(self):
        self.queries = []

    async def ainvoke(self, state):
        self.queries.append(state["query"])
        if "Exclude TSLA" not in state["query"]:
            return {"status": "clarify", "answer": "Which four assets should I use?"}
        if "2025-01-01" not in state["query"]:
            return {"status": "clarify", "answer": "Please supply from/to dates."}
        return {"status": "answered", "answer": "AAPL, MSFT, META and NVDA comparison"}


async def test_market_resumes_in_domain_and_retains_multiple_replies():
    classifier, agent = Classifier(), MarketAgent()
    graph = build_graph(classifier, market_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "market-resume"}}
    paused = await graph.ainvoke(
        {"query": "Compare AAPL, MSFT, META, NVDA and TSLA growth in Q2"}, config
    )
    assert paused["__interrupt__"][0].value["kind"] == "clarification"
    assert classifier.calls == 1
    assert len(agent.queries) == 1
    paused_again = await graph.ainvoke(Command(resume="Exclude TSLA"), config)
    assert paused_again["__interrupt__"][0].value["question"] == "Please supply from/to dates."
    assert len(agent.queries) == 2
    result = await graph.ainvoke(Command(resume="from 2025-01-01 to 2025-03-31"), config)
    assert result["status"] == "answered"
    assert result["executed_handler"] == "market_analysis_node"
    assert classifier.calls == 1
    assert len(agent.queries) == 3
    assert "Exclude TSLA" in agent.queries[-1]
    assert "2025-03-31" in agent.queries[-1]
    assert result["clarification_question"] == ""


def test_market_uses_existing_resume_api():
    classifier, agent = Classifier(), MarketAgent()
    graph = build_graph(classifier, market_agent=agent, checkpointer=MemorySaver())
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph
    with TestClient(app) as client:
        paused = client.post(
            "/query",
            json={"query": "Compare AAPL, MSFT, META, NVDA and TSLA from 2025-01-01 to 2025-03-31"},
        ).json()
        assert paused["status"] == "clarification"
        assert paused["thread_id"]
        assert paused["clarification_question"] == "Which four assets should I use?"
        result = client.post(
            "/query/resume", json={"thread_id": paused["thread_id"], "reply": "Exclude TSLA"}
        ).json()
    assert result["intent"] == "market"
    assert result["executed_handler"] == "market_analysis_node"
    assert "AAPL, MSFT, META and NVDA" in result["answer"]
    assert "thread_id" not in result
    assert "clarification_question" not in result
    assert classifier.calls == 1


async def test_real_market_agent_resumes_exclusion(monkeypatch):
    from langchain_core.messages import AIMessage
    from mcp.types import CallToolResult, TextContent

    from test_market_analysis_agent import make_agent

    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(content="Please choose up to four assets."),
            AIMessage(
                content="",
                tool_calls=[{"name": "resolve_market_timeframe", "args": {}, "id": "window"}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_price_growth",
                        "args": {"symbol": ["AAPL", "MSFT", "META", "NVDA"]},
                        "id": "growth",
                    }
                ],
            ),
        ],
    )

    async def fetch(name, arguments):
        return CallToolResult(
            content=[TextContent(type="text", text=f"datetime;close\n{arguments['end_date']};100")]
        )

    session.call_tool.side_effect = fetch
    classifier = Classifier()
    graph = build_graph(classifier, market_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "real-market-resume"}}
    paused = await graph.ainvoke(
        {"query": "Compare AAPL, MSFT, META, NVDA and TSLA growth from 2025-01-01 to 2025-02-01"},
        config,
    )
    assert paused["__interrupt__"][0].value["question"] == "Please choose up to four assets."
    session.call_tool.assert_not_awaited()
    result = await graph.ainvoke(Command(resume="Exclude TSLA"), config)
    assert result["status"] == "answered"
    assert [item["symbol"] for item in result["calculations"]] == ["AAPL", "MSFT", "META", "NVDA"]
    assert classifier.calls == 1
