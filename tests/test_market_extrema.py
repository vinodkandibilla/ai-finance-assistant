from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from mcp.types import CallToolResult, TextContent

from finassist.agents.market_analysis import extrema
from finassist.agents.market_analysis.extrema import compute_price_extrema
from test_market_analysis_agent import make_agent

ROWS = [
    {"datetime": "2026-10-01", "high": "120.25", "low": "100", "close": "110"},
    {"datetime": "2026-10-02", "high": "125.50", "low": "90", "close": "105"},
    {"datetime": "2026-10-03", "high": "125.50", "low": "95", "close": "115"},
]


@pytest.mark.parametrize(
    "operation,field,value,dates",
    [
        ("max", "high", "125.50", ["2026-10-02", "2026-10-03"]),
        ("min", "low", "90", ["2026-10-02"]),
        ("max", "close", "115", ["2026-10-03"]),
    ],
)
def test_extrema_fields_and_ties(operation, field, value, dates):
    result = compute_price_extrema(ROWS, "2026-10-01", "2026-10-03", operation, field)
    assert result["value"] == value
    assert result["observation_dates"] == dates
    assert result["coverage"] == "unverified"


@pytest.mark.parametrize("price", ["NaN", "Infinity", "-1", "0", "invalid"])
def test_invalid_prices_fail_closed(price):
    with pytest.raises(ValueError):
        compute_price_extrema(
            [{"datetime": "2026-10-01", "high": price}], "2026-10-01", "2026-10-03", "max", "high"
        )


def test_window_filter_and_empty_history():
    result = compute_price_extrema(ROWS, "2026-10-01", "2026-10-01", "max", "high")
    assert result["value"] == "120.25"
    with pytest.raises(ValueError):
        compute_price_extrema([], "2026-10-01", "2026-10-03", "max", "high")


async def test_agent_calls_deterministic_extrema_tool(monkeypatch):
    calls = [
        {"name": "resolve_market_timeframe", "args": {}, "id": "window"},
        {
            "name": "calculate_price_extrema",
            "args": {"symbol": "TSLA", "operation": "max"},
            "id": "maximum",
        },
    ]
    agent, _, _ = make_agent(
        monkeypatch, [AIMessage(content="", tool_calls=[call]) for call in calls]
    )
    cache = AsyncMock()
    cache.call_tool.return_value = CallToolResult(
        content=[
            TextContent(
                type="text",
                text='{"meta":{"symbol":"TSLA","currency":"USD"},"values":'
                '[{"datetime":"2026-10-01","close":"110","high":"120.25"}]}',
            )
        ]
    )
    monkeypatch.setattr(extrema, "market_cache", cache)
    result = await agent.ainvoke({"query": "TSLA highest price from 2026-10-01 to 2026-10-03"})
    assert result["status"] == "answered"
    assert result["calculation"]["value"] == "120.25"
    assert result["calculation"]["field"] == "high"
    assert cache.call_tool.call_args.args[1] == "get_time_series"
    assert cache.call_tool.call_args.args[2]["interval"] == "1day"


def test_extrema_reject_conflicting_duplicates_and_invalid_ohlc():
    for rows in [
        [ROWS[0], {**ROWS[0], "high": "200"}],
        [{"datetime": "2026-10-01", "open": "150", "close": "110", "high": "120", "low": "100"}],
    ]:
        with pytest.raises(ValueError):
            compute_price_extrema(rows, "2026-10-01", "2026-10-03", "max", "high")
