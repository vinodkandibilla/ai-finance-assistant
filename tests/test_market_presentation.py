import json

import pytest
from langchain_core.messages import AIMessage
from mcp.types import CallToolResult, TextContent

from finassist.agents.market_analysis.presentation import compact_tool_result, format_snapshot
from test_market_analysis_agent import make_agent


def provider(data):
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))])


async def test_routine_quote_finishes_without_final_llm_turn(monkeypatch):
    agent, session, model = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "q",
                    }
                ],
            )
        ],
    )
    session.call_tool.return_value = provider(
        {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "currency": "USD",
            "close": "200",
            "datetime": "2026-10-20 12:00:00",
            "volume": "1000",
            "percent_change": "1.5",
        }
    )
    result = await agent.ainvoke({"query": "AAPL quote"})
    assert result["status"] == "answered"
    assert "USD 200" in result["answer"]
    assert "2026-10-20 12:00:00" in result["answer"]
    assert "may be cached" in result["answer"]
    assert result["calculations"][0]["values"][0]["volume"] == "1000"
    session.call_tool.assert_awaited_once_with("get_quote", {"symbol": "AAPL"})
    quote_schema = next(
        d["function"] for d in model.descriptors if d["function"]["name"] == "get_quote"
    )
    assert "finish_response" in quote_schema["parameters"]["properties"]
    assert "finish_response" in quote_schema["parameters"]["required"]
    history_schema = next(
        d["function"] for d in model.descriptors if d["function"]["name"] == "get_time_series"
    )
    assert "finish_response" in history_schema["parameters"]["required"]
    # Local additions must never mutate the cached MCP catalog.
    assert all(
        "finish_response" not in tool.inputSchema.get("properties", {})
        for tool in session.list_tools.return_value.tools
    )


async def test_daily_snapshot_keeps_quote_and_full_chart_without_final_turn(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "q"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_time_series",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "h",
                    }
                ],
            ),
        ],
    )
    rows = [
        {
            "datetime": "2026-10-01",
            "open": "99",
            "high": "102",
            "low": "98",
            "close": "100",
            "volume": "500",
        },
        {"datetime": "2026-10-02", "close": "101"},
    ]
    session.call_tool.side_effect = [
        provider({"symbol": "AAPL", "currency": "USD", "close": "105"}),
        provider({"meta": {"symbol": "AAPL"}, "values": rows}),
    ]
    result = await agent.ainvoke({"query": "AAPL quote and daily chart over the last month"})
    assert result["status"] == "answered"
    assert "2 provider observations" in result["answer"]
    assert result["calculations"][1]["values"] == rows
    assert result["calculations"][1]["currency"] == "USD"
    args = session.call_tool.await_args_list[1].args[1]
    assert "finish_response" not in args
    assert args["interval"] == "1day"


async def test_error_with_finish_flag_still_goes_back_to_llm(monkeypatch):
    agent, _, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "q",
                    }
                ],
            ),
            AIMessage(content="Quote unavailable; please check the symbol."),
        ],
        error=True,
    )
    result = await agent.ainvoke({"query": "AAPL quote"})
    assert result["answer"] == "Quote unavailable; please check the symbol."
    assert not result.get("calculations")


async def test_analytical_request_keeps_llm_narration_and_compact_data(monkeypatch):
    agent, session, model = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "q"}],
            ),
            AIMessage(content="Analysis of the supplied quote."),
        ],
    )
    session.call_tool.return_value = provider({"close": "100", "currency": "USD"})
    result = await agent.ainvoke({"query": "Explain the quote fields for AAPL"})
    assert result["answer"] == "Analysis of the supplied quote."
    assert json.loads(model.messages[-1].content) == {"close": "100", "currency": "USD"}
    assert "content" not in json.loads(model.messages[-1].content)


def test_history_compaction_preserves_all_rows_and_fields():
    rows = [
        {"datetime": str(i), "open": str(i), "close": str(i + 1), "volume": "500"}
        for i in range(100)
    ]
    original = provider({"meta": {"timezone": "America/New_York"}, "values": rows})
    compact = compact_tool_result(original)
    decoded = json.loads(compact)
    assert [
        dict(zip(decoded["columns"], values, strict=True)) for values in decoded["rows"]
    ] == rows
    assert decoded["meta"]["timezone"] == "America/New_York"
    assert len(compact) < len(json.dumps(original.model_dump(mode="json")))


def test_missing_time_is_explicit_not_invented():
    answer = format_snapshot(
        [{"symbol": "AAPL", "interval": "quote", "values": [{"close": "100"}]}]
    )
    assert "observation time: unavailable" in answer
    assert "currency unavailable" in answer


def test_unstructured_error_text_is_retained():
    result = CallToolResult(content=[TextContent(type="text", text="Provider unavailable")])
    assert json.loads(compact_tool_result(result)) == "Provider unavailable"


async def test_batch_error_prevents_python_completion(monkeypatch):
    from finassist.agents.market_analysis.cache import market_cache

    monkeypatch.setattr(market_cache, "max_retries", 0)
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_quote", "args": {"symbol": "MSFT"}, "id": "bad"},
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "ok",
                    },
                ],
            ),
            AIMessage(content="AAPL is available, but MSFT retrieval failed."),
        ],
    )
    session.call_tool.side_effect = [
        CallToolResult(isError=True, content=[TextContent(type="text", text="Provider error")]),
        provider({"symbol": "AAPL", "close": "100"}),
    ]
    result = await agent.ainvoke({"query": "AAPL and MSFT quotes"})
    assert result["answer"] == "AAPL is available, but MSFT retrieval failed."
    assert session.call_tool.await_count == 2


def test_csv_history_compaction_keeps_all_fields():
    result = CallToolResult(
        content=[TextContent(type="text", text="datetime;close;volume\n2026-10-01;100;500")]
    )
    data = json.loads(compact_tool_result(result))
    assert data["columns"] == ["datetime", "close", "volume"]
    assert data["rows"] == [["2026-10-01", "100", "500"]]


async def test_single_turn_reordered_batch_finishes_snapshot(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_time_series", "args": {"symbol": "AAPL"}, "id": "history"},
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "quote",
                    },
                ],
            )
        ],
    )
    session.call_tool.side_effect = [
        provider(
            {"meta": {"symbol": "AAPL"}, "values": [{"datetime": "2026-10-01", "close": "100"}]}
        ),
        provider({"symbol": "AAPL", "currency": "USD", "close": "105"}),
    ]
    result = await agent.ainvoke({"query": "AAPL quote and daily chart over the last month"})
    assert result["status"] == "answered"
    assert len(result["calculations"]) == 2
    assert "USD 105" in result["answer"]
    assert [call.args[0] for call in session.call_tool.await_args_list] == [
        "get_time_series",
        "get_quote",
    ]
    assert "start_date" in session.call_tool.await_args_list[0].args[1]


async def test_batched_quarter_clarification_blocks_all_provider_calls(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "quote"},
                    {
                        "name": "get_time_series",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "h",
                    },
                ],
            )
        ],
    )
    result = await agent.ainvoke({"query": "AAPL quote and chart for Q2"})
    assert result["status"] == "clarify"
    assert result["clarification_question"]
    session.call_tool.assert_not_awaited()


async def test_same_turn_growth_batch_preserves_all_symbols(monkeypatch):
    from unittest.mock import AsyncMock

    from finassist.agents.market_analysis import agent as module

    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculate_price_growth", "args": {"symbol": "AAPL"}, "id": "a"},
                    {"name": "calculate_price_growth", "args": {"symbol": "MSFT"}, "id": "m"},
                ],
            )
        ],
    )
    calculate = AsyncMock(return_value={"status": "answered", "answer": "Comparison"})
    monkeypatch.setattr(module, "calculate_price_growth", calculate)
    result = await agent.ainvoke({"query": "Compare AAPL and MSFT growth over the last month"})
    assert result["status"] == "answered"
    assert calculate.await_args.args[1] == ["AAPL", "MSFT"]
    assert "start_date" in calculate.await_args.args[2]
    session.call_tool.assert_not_awaited()


async def test_aap_remains_distinct_from_aapl_without_exchange_scope(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAP", "finish_response": True},
                        "id": "q",
                    }
                ],
            )
        ],
    )
    session.call_tool.return_value = provider({"symbol": "AAP", "exchange": "NYSE", "close": "44"})
    result = await agent.ainvoke({"query": "AAP quote"})
    assert result["status"] == "answered"
    assert "AAP (NYSE)" in result["answer"]
    assert result["calculations"][0]["symbol"] == "AAP"


@pytest.mark.parametrize("structured", [False, True])
async def test_market_page_cannot_substitute_nyse_for_nasdaq(monkeypatch, structured):
    from finassist.web_app.services.showcase import scope_market_question

    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_quote",
                        "args": {
                            "symbol": "AAP",
                            "exchange": "NYSE",
                            "mic_code": "XNYS",
                            "finish_response": True,
                        },
                        "id": "q",
                    }
                ],
            )
        ],
    )
    session.call_tool.return_value = provider({"symbol": "AAP", "exchange": "NYSE", "close": "44"})
    state = (
        {"query": "AAP quote", "market_scope": "us_nasdaq_usd"}
        if structured
        else {"query": scope_market_question("AAP quote")}
    )
    result = await agent.ainvoke(state)
    assert result["status"] == "error"
    assert "NASDAQ listing" in result["answer"]
    assert not result.get("calculations")
    session.call_tool.assert_awaited_once_with(
        "get_quote", {"symbol": "AAP", "exchange": "NASDAQ", "country": "United States"}
    )


async def test_quote_only_never_resolves_timeframe(monkeypatch):
    from unittest.mock import AsyncMock

    from finassist.agents.market_analysis import agent as module

    resolver = AsyncMock(side_effect=AssertionError("Quote needs no timeframe"))
    from types import SimpleNamespace

    monkeypatch.setattr(module, "resolve_market_timeframe", SimpleNamespace(ainvoke=resolver))
    agent, _, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "q",
                    }
                ],
            )
        ],
    )
    assert (await agent.ainvoke({"query": "AAPL quote"}))["status"] == "answered"
    resolver.assert_not_awaited()


async def test_structured_market_scope_rejects_non_usd_observations(monkeypatch):
    agent, session, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_quote",
                        "args": {"symbol": "AAPL", "finish_response": True},
                        "id": "q",
                    }
                ],
            )
        ],
    )
    session.call_tool.return_value = provider(
        {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "currency": "EUR",
            "close": "44",
        }
    )
    result = await agent.ainvoke({"query": "AAPL quote", "market_scope": "us_nasdaq_usd"})
    assert result["status"] == "error"
    assert "requires USD" in result["answer"]
    assert not result.get("calculations")
