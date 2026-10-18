import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from finassist.api.query import get_query_graph
from finassist.main import create_app
from finassist.workflow.builder import build_graph
from test_finance_autonomous_agent import call
from test_news_synthesizer_agent import evidence, make_agent


class Classifier:
    def __init__(self, intent="news"):
        self.intent = intent
        self.calls = []

    async def classify(self, query):
        self.calls.append(query)
        return self.intent


@pytest.mark.parametrize(
    "query",
    [
        "Give me the latest financial-market news.",
        "Give me the latest Apple news.",
        "Summarize news affecting semiconductor stocks.",
    ],
)
def test_scoped_news_reaches_agent_without_generic_clarification(query):
    classifier = Classifier()
    agent, _, calls = make_agent(
        [call("search_financial_news", {"query": query}, "search")], batches=[evidence()]
    )
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: build_graph(classifier, news_agent=agent)
    with TestClient(app) as client:
        body = client.post("/query", json={"query": query}).json()
    assert body["intent"] == body["classified_intent"] == "news"
    assert body["executed_handler"] == "news_synthesizer_node"
    assert body["citations"] == ["https://example.com/story?b=2&a=1"]
    assert "thread_id" not in body
    assert calls == [query]


@pytest.mark.parametrize(
    "query,reply",
    [
        ("Give me the latest news.", "Financial markets."),
        ("How can higher interest rates affect technology stocks?", "Yes, recent news"),
    ],
)
def test_news_clarification_resumes_same_agent_and_keeps_context(query, reply):
    classifier = Classifier()
    agent, model, calls = make_agent(
        [
            call("ask_news_clarification", {"question": "Which news topic would you like?"}, "ask"),
            call("search_financial_news", {"query": "financial-market news"}, "search"),
        ],
        batches=[evidence()],
    )
    graph = build_graph(classifier, news_agent=agent, checkpointer=MemorySaver())
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph
    with TestClient(app) as client:
        paused = client.post("/query", json={"query": query}).json()
        assert paused["status"] == "clarification"
        assert paused["executed_handler"] == "news_synthesizer_node"
        assert calls == []
        completed = client.post(
            "/query/resume",
            json={
                "thread_id": paused["thread_id"],
                "reply": reply,
            },
        ).json()
    assert completed["intent"] == "news"
    assert completed["executed_handler"] == "news_synthesizer_node"
    assert "thread_id" not in completed
    assert classifier.calls == [query]
    assert any(m.content == query for m in model.seen[-1])


@pytest.mark.parametrize(
    "query", ["What is AAPL's current price?", "Show AAPL history for 5 days."]
)
async def test_market_classification_does_not_invoke_news(query):
    class Market:
        async def ainvoke(self, state):
            return {"answer": "Market data", "status": "answered"}

    agent, model, calls = make_agent([AIMessage(content="Must not run")])
    graph = build_graph(Classifier("market"), market_agent=Market(), news_agent=agent)
    result = await graph.ainvoke({"query": query})
    assert result["executed_handler"] == "market_analysis_node"
    assert calls == []
    assert not model.seen
