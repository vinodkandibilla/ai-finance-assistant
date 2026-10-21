from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from pydantic import SecretStr

from finassist.agents.market_analysis import agent as module
from finassist.agents.market_analysis.agent import MarketAnalysisAgent
from finassist.core.config import Settings
from finassist.workflow.builder import build_graph


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.descriptors = []
        self.messages = []

    def bind_tools(self, descriptors):
        self.descriptors = descriptors
        return self

    async def ainvoke(self, messages):
        self.messages = list(messages)
        return next(self.responses)


def make_agent(monkeypatch, responses, *, error=False, tools=None):
    session = AsyncMock()
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(name=name, inputSchema={"type": "object", "properties": {}})
            for name in (tools or ["get_quote", "search_symbol", "get_time_series"])
        ]
    )
    session.call_tool.return_value = CallToolResult(
        isError=error, content=[TextContent(type="text", text='{"close":"212.35"}')]
    )

    @asynccontextmanager
    async def connection(settings):
        yield session

    monkeypatch.setattr(module, "twelve_data_session", connection)
    model = FakeModel(responses)
    settings = Settings(_env_file=None, twelve_data_api_key=SecretStr("test-key"))
    return MarketAnalysisAgent(llm=model, settings=settings), session, model


async def test_quote_tool_conversation_and_allowlist(monkeypatch):
    agent, session, model = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "QQQ"}, "id": "quote-1"}],
            ),
            AIMessage(content="QQQ: USD 212.35. Volume unavailable; daily change unavailable."),
        ],
    )
    result = await agent.ainvoke({"query": "Quote QQQ"})
    assert result["status"] == "answered"
    assert result["calculations"][0]["type"] == "market_observations"
    assert result["calculations"][0]["symbol"] == "QQQ"
    assert result["calculations"][0]["values"][0]["close"] == "212.35"
    session.call_tool.assert_awaited_once_with("get_quote", {"symbol": "QQQ"})
    assert {tool["function"]["name"] for tool in model.descriptors} == {
        "get_quote",
        "search_symbol",
        "get_time_series",
        "calculate_price_growth",
        "calculate_price_extrema",
    }
    assert model.messages[-1].tool_call_id == "quote-1"


async def test_disabled_and_missing_key_do_not_open_connection(monkeypatch):
    connection = AsyncMock(side_effect=AssertionError("Must not connect"))
    monkeypatch.setattr(module, "twelve_data_session", connection)
    for settings, expected in [
        (Settings(_env_file=None, market_quotes_enabled=False), "Market quotes are disabled."),
        (Settings(_env_file=None), "Market quotes are unavailable. Set TWELVE_DATA_API_KEY."),
    ]:
        result = await MarketAnalysisAgent(llm=None, settings=settings).ainvoke({"query": "QQQ"})
        assert result["answer"] == expected
    connection.assert_not_called()


async def test_mcp_failure_does_not_expose_credentials(monkeypatch):
    agent, session, _ = make_agent(monkeypatch, [])
    session.list_tools.side_effect = RuntimeError("secret-key in provider URL")
    result = await agent.ainvoke({"query": "AAPL"})
    assert result == {"answer": module.UNAVAILABLE, "status": "error"}


async def test_missing_quote_tool(monkeypatch):
    agent, session, _ = make_agent(monkeypatch, [], tools=["search_symbol"])
    result = await agent.ainvoke({"query": "AAPL"})
    assert result["status"] == "error"
    session.call_tool.assert_not_awaited()


async def test_unknown_tool_never_executes(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_financials", "args": {"symbol": "AAPL"}, "id": "bad"}],
            ),
            AIMessage(content="History is not implemented yet."),
        ],
    )
    await agent.ainvoke({"query": "AAPL history"})
    session.call_tool.assert_not_awaited()


async def test_graph_uses_new_agent_for_qqq(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "QQQ"}, "id": "quote"}],
            ),
            AIMessage(content="QQQ quote"),
        ],
    )

    class Classifier:
        async def classify(self, query):
            return "market"

    result = await build_graph(Classifier(), market_agent=agent).ainvoke(
        {"query": "What is the stock price of QQQ?"}
    )
    assert result["executed_handler"] == "market_analysis_node"
    assert result["answer"] == "QQQ quote"
    session.call_tool.assert_awaited_once()


async def test_symbol_search_before_quote(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_symbol", "args": {"symbol": "Apple"}, "id": "search"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "quote"}],
            ),
            AIMessage(content="AAPL quote"),
        ],
    )
    await agent.ainvoke({"query": "Apple quote"})
    assert [call.args[0] for call in session.call_tool.await_args_list] == [
        "search_symbol",
        "get_quote",
    ]


async def test_tool_error_is_passed_to_model(monkeypatch):
    agent, _, model = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "QQQ"}, "id": "quote"}],
            ),
            AIMessage(content="Quote unavailable."),
        ],
        error=True,
    )
    result = await agent.ainvoke({"query": "QQQ quote"})
    assert result["answer"] == "Quote unavailable."
    assert '"isError": true' in model.messages[-1].content


async def test_deadline_cancels_slow_server(monkeypatch):
    import asyncio

    agent, session, _ = make_agent(monkeypatch, [])
    agent.settings.market_agent_timeout_seconds = 0.01

    async def slow_catalog():
        await asyncio.sleep(1)

    session.list_tools.side_effect = slow_catalog
    result = await agent.ainvoke({"query": "AAPL"})
    assert result["status"] == "error"
    assert "time limit" in result["answer"]


async def test_history_uses_deterministic_parameters(monkeypatch):
    agent, session, model = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_time_series",
                        "args": {
                            "symbol": "QQQ",
                            "interval": "1day",
                            "outputsize": 999,
                            "start_date": "1900-01-01",
                        },
                        "id": "history",
                    }
                ],
            ),
            AIMessage(content="QQQ weekly closing prices"),
        ],
    )
    result = await agent.ainvoke({"query": "QQQ weekly data for the last 12 weeks"})
    assert result["status"] == "answered"
    session.call_tool.assert_awaited_once_with(
        "get_time_series", {"symbol": "QQQ", "interval": "1week", "outputsize": 12}
    )


async def test_history_automatically_requests_missing_timeframe(monkeypatch):
    agent, session, model = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_time_series", "args": {"symbol": "AAPL"}, "id": "history"}
                ],
            ),
            AIMessage(content="Please specify a timeframe."),
        ],
    )
    result = await agent.ainvoke({"query": "AAPL history"})
    session.call_tool.assert_not_awaited()
    assert result["status"] == "clarify"
    assert "from/to dates" in result["clarification_question"]


async def test_unknown_fiscal_period_stops_before_provider_call(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_time_series", "args": {"symbol": "AAPL"}, "id": "history"}
                ],
            ),
        ],
    )
    result = await agent.ainvoke({"query": "AAPL fiscal Q2 results 2025"})
    assert result["status"] == "clarify"
    assert "explicit from/to dates" in result["answer"]
    session.call_tool.assert_not_awaited()


async def test_search_requests_are_capped_at_one_hundred(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_symbol",
                        "args": {"symbol": "Apple", "outputsize": 5000},
                        "id": "search",
                    }
                ],
            ),
            AIMessage(content="AAPL"),
        ],
    )
    session.list_tools.return_value.tools[1].inputSchema = {
        "type": "object",
        "properties": {"outputsize": {"type": "integer"}},
    }
    await agent.ainvoke({"query": "Find Apple"})
    session.call_tool.assert_awaited_once_with(
        "search_symbol", {"symbol": "Apple", "outputsize": 100}
    )


async def test_growth_uses_provider_prices_and_returns_python_math(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_price_growth",
                        "args": {"symbol": "AAPL", "start_price": 999},
                        "id": "growth",
                    }
                ],
            ),
        ],
    )
    session.call_tool.side_effect = [
        CallToolResult(content=[TextContent(type="text", text="datetime;close\n2025-01-01;100")]),
        CallToolResult(content=[TextContent(type="text", text="datetime;close\n2025-02-01;125")]),
    ]
    result = await agent.ainvoke({"query": "AAPL price growth from 2025-01-01 to 2025-02-01"})
    assert result["status"] == "answered"
    assert result["calculation"]["start_price"] == "100"
    assert "(125 - 100) / 100 × 100 = 25.00%" in result["answer"]


async def test_multiple_growth_calls_include_both_assets(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculate_price_growth", "args": {"symbol": "AAPL"}, "id": "a"},
                    {"name": "calculate_price_growth", "args": {"symbol": "MSFT"}, "id": "b"},
                ],
            ),
        ],
    )

    async def fetch(name, arguments):
        price = "100" if arguments["end_date"] == "2025-01-01" else "110"
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"datetime;close\n{arguments['end_date']};{price}")
            ]
        )

    session.call_tool.side_effect = fetch
    result = await agent.ainvoke(
        {"query": "Compare AAPL and MSFT growth from 2025-01-01 to 2025-02-01"}
    )
    assert result["status"] == "answered"
    assert [item["symbol"] for item in result["calculations"]] == ["AAPL", "MSFT"]


async def test_quote_rate_limit_returns_retry_message(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "quote"}],
            )
        ],
    )
    session.call_tool.return_value = CallToolResult(
        isError=True, content=[TextContent(type="text", text="429: Too many requests")]
    )
    result = await agent.ainvoke({"query": "AAPL quote"})
    assert result["status"] == "error"
    assert "try again after one minute" in result["answer"]
