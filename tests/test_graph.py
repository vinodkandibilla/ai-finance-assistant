import pytest

from ai_finance_assistant.graph.builder import (
    CLARIFY_RESPONSE,
    MARKET_QUOTE_UNAVAILABLE_RESPONSE,
    NEWS_RESPONSE,
    PORTFOLIO_ANALYSIS_RESPONSE,
    build_graph,
)
from ai_finance_assistant.graph.state import FinanceIntent


class FakeClassifier:
    def __init__(self, intent: FinanceIntent) -> None:
        self.intent = intent

    async def classify(self, query: str) -> FinanceIntent:
        return self.intent


async def test_graph_routes_financial_education_to_empty_finance_qa_node() -> None:
    result = await build_graph(FakeClassifier("finance_qa")).ainvoke(
        {"query": "Explain how an ETF works."}
    )

    assert result["intent"] == "finance_qa"
    assert result["answer"] == ""


class FakeMarketAgent:
    def __init__(self):
        self.queries = []

    async def ainvoke(self, state):
        self.queries.append(state["query"])
        return {"answer": "Market agent response", "status": "answered"}


async def test_graph_routes_other_intent_to_unsupported_node() -> None:
    result = await build_graph(FakeClassifier("other")).ainvoke({"query": "What is an ETF?"})

    assert result["intent"] == "other"
    assert result["answer"] == "This request is not supported yet."


async def test_graph_routes_portfolio_intent_to_computational_node() -> None:
    result = await build_graph(FakeClassifier("portfolio")).ainvoke(
        {"query": "Analyze my portfolio allocation and concentration."}
    )

    assert result["intent"] == "portfolio"
    assert result["answer"] == PORTFOLIO_ANALYSIS_RESPONSE


@pytest.mark.parametrize("symbol", ["AAPL", "IBM", "QQQ"])
async def test_graph_routes_market_intent_to_market_analysis_node(symbol):
    agent = FakeMarketAgent()
    query = f"What is the current price of {symbol}?"
    result = await build_graph(FakeClassifier("market"), market_agent=agent).ainvoke(
        {"query": query}
    )
    assert result["intent"] == "market"
    assert result["answer"] == "Market agent response"
    assert agent.queries == [query]


async def test_graph_returns_unavailable_response_when_quote_tool_fails() -> None:
    result = await build_graph(FakeClassifier("market"), market_agent=None).ainvoke(
        {"query": "What is the current price of AAPL?"}
    )

    assert result["answer"] == MARKET_QUOTE_UNAVAILABLE_RESPONSE


async def test_graph_routes_goals_intent_to_computational_node() -> None:
    result = await build_graph(FakeClassifier("goals")).ainvoke(
        {"query": "How much should I save each month?"}
    )

    assert result["intent"] == "goals"
    assert result["answer"] == "Goal planning agent is not configured."
    assert result["status"] == "error"


async def test_graph_routes_news_intent_to_news_synthesizer_node() -> None:
    result = await build_graph(FakeClassifier("news")).ainvoke(
        {"query": "Give me a concise summary of today's top AI news."}
    )

    assert result["intent"] == "news"
    assert result["answer"] == NEWS_RESPONSE


async def test_graph_routes_tax_intent_to_tax_education_node() -> None:
    result = await build_graph(FakeClassifier("tax")).ainvoke(
        {
            "query": "What is the standard deduction for Married Filing Jointly "
            "with two qualifying children for the year 2026?"
        }
    )

    assert result["intent"] == "tax"
    assert result["answer"] == "Tax education agent is not configured."
    assert result["status"] == "error"


async def test_graph_routes_market_query_without_ticker_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("market")).ainvoke(
        {"query": "What is the stock price?"}
    )

    assert result["intent"] == "market"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_market_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("market")).ainvoke(
        {"query": "What is the stock price?"}
    )

    assert result["intent"] == "market"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_news_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("news")).ainvoke({"query": "top news"})

    assert result["intent"] == "news"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_or_incomplete_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("clarify")).ainvoke(
        {"query": "What is the stock price?"}
    )

    assert result["intent"] == "clarify"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_out_of_scope_stops_without_invoking_another_domain():
    from types import SimpleNamespace

    from langgraph.graph import END, START, StateGraph

    from ai_finance_assistant.graph.state import FinanceAssistantState

    def finance(state):
        return {
            "status": "out_of_scope",
            "answer": "Please submit a new question.",
            "citations": [],
        }

    def tax(state):
        raise AssertionError("Domain agents must not dispatch to other agents")

    def agent(node):
        workflow = StateGraph(FinanceAssistantState)
        workflow.add_node("run", node)
        workflow.add_edge(START, "run")
        workflow.add_edge("run", END)
        return SimpleNamespace(graph=workflow.compile())

    graph = build_graph(
        FakeClassifier("finance_qa"), finance_agent=agent(finance), tax_agent=agent(tax)
    )
    result = await graph.ainvoke({"query": "What is an ETF?"})
    assert result["intent"] == "finance_qa"
    assert result["status"] == "out_of_scope"
    assert not result["citations"]


@pytest.mark.parametrize(
    "query",
    [
        "Explain compound interest while I save for retirement.",
        "Explain ETF expense ratios; I will consider tax later.",
        "Explain diversification after a payroll deduction funds my investments.",
    ],
)
async def test_incidental_tax_words_do_not_override_finance_classification(query):
    result = await build_graph(FakeClassifier("finance_qa")).ainvoke({"query": query})
    assert result["classified_intent"] == "finance_qa"
    assert result["executed_handler"] == "finance_qa_node"
    assert result["status"] == "not_implemented"


async def test_tax_classification_still_routes_without_a_tax_keyword():
    result = await build_graph(FakeClassifier("tax")).ainvoke({"query": "Explain wash-sale rules"})
    assert result["classified_intent"] == "tax"
    assert result["executed_handler"] == "tax_education_node"


async def test_disabled_market_feature_returns_disabled_without_invoking_tool(monkeypatch):
    from ai_finance_assistant.core.config import Settings
    from ai_finance_assistant.graph import builder

    monkeypatch.setattr(
        builder, "get_settings", lambda: Settings(_env_file=None, market_quotes_enabled=False)
    )
    agent = FakeMarketAgent()
    for query in ("What is AAPL's price?", "What is the stock price?", "What is IBM's price?"):
        result = await build_graph(FakeClassifier("market"), market_agent=agent).ainvoke(
            {"query": query}
        )
        assert result["answer"] == "Market quotes are disabled."
        assert result["classified_intent"] == "market"
        assert result["executed_handler"] == "market_analysis_node"
    assert not agent.queries
