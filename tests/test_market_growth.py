from datetime import date
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from finassist.agents.market_analysis.growth import (
    calculate_price_growth,
    compute_price_growth,
)
from finassist.agents.market_analysis.timeframe import resolve_timeframe


def response(text):
    return CallToolResult(content=[TextContent(type="text", text=text)])


def test_six_month_growth_uses_calendar_dates():
    window = resolve_timeframe("AAPL price growth in the last 6 months", date(2026, 10, 13))
    assert window["parameters"]["start_date"] == "2026-04-13"
    assert window["parameters"]["end_date"] == "2026-10-13"
    assert window["parameters"]["interval"] == "1day"


@pytest.mark.parametrize(
    "start,end,percent",
    [("100", "125", "25.00"), ("100", "75", "-25.00"), ("0.1", "0.3", "200.00")],
)
def test_growth_arithmetic(start, end, percent):
    result = compute_price_growth(
        {"close": start, "datetime": "2025-01-01"}, {"close": end, "datetime": "2025-03-31"}
    )
    assert result["math"].endswith(percent + "%")


@pytest.mark.parametrize("price", ["0", "-1", "NaN", "Infinity"])
def test_invalid_baselines(price):
    with pytest.raises(ValueError):
        compute_price_growth(
            {"close": price, "datetime": "2025-01-01"}, {"close": "10", "datetime": "2025-02-01"}
        )


async def test_weekend_baseline_and_historical_end():
    session = AsyncMock()
    session.call_tool.side_effect = [
        response("datetime;close\n2025-01-06;999\n2025-01-03;100\n2025-01-02;90"),
        response('{"values":[{"datetime":"2025-03-28","close":"125"}]}'),
    ]
    result = await calculate_price_growth(
        session, "AAPL", {"start_date": "2025-01-05", "end_date": "2025-03-30"}, "2026-10-13"
    )
    assert result["status"] == "answered"
    assert result["calculation"]["start_date"] == "2025-01-03"
    assert result["calculation"]["end_date"] == "2025-03-28"
    assert "25.00%" in result["answer"]
    assert all(call.args[1]["outputsize"] <= 100 for call in session.call_tool.await_args_list)


async def test_current_endpoint_uses_quote():
    session = AsyncMock()
    session.call_tool.side_effect = [
        response("datetime;close\n2026-04-13;100"),
        response("symbol;datetime;close\nAAPL;2026-10-13;110"),
    ]
    result = await calculate_price_growth(
        session, "AAPL", {"start_date": "2026-04-13", "end_date": "2026-10-13"}, "2026-10-13"
    )
    assert result["status"] == "answered"
    assert "10.00%" in result["answer"]
    session.call_tool.assert_awaited_with("get_quote", {"symbol": "AAPL"})


async def test_missing_baseline_does_not_compute():
    session = AsyncMock()
    session.call_tool.return_value = response("datetime;close\n2025-01-06;100")
    result = await calculate_price_growth(
        session, "AAPL", {"start_date": "2025-01-05", "end_date": "2025-03-30"}, "2026-10-13"
    )
    assert result["status"] == "error"
    assert "calculation" not in result


async def test_comparison_fetches_sequentially_and_aligns_dates():
    import asyncio

    session = AsyncMock()
    active = 0
    peak = 0

    async def fetch(name, arguments):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        if arguments["end_date"] == "2025-01-05":
            if arguments["symbol"] == "AAPL":
                return response("datetime;close\n2025-01-03;100\n2025-01-02;90")
            return response("datetime;close\n2025-01-02;200")
        if arguments["symbol"] == "AAPL":
            return response("datetime;close\n2025-03-28;108")
        return response("datetime;close\n2025-03-28;220")

    session.call_tool.side_effect = fetch
    result = await calculate_price_growth(
        session,
        ["AAPL", "MSFT"],
        {"start_date": "2025-01-05", "end_date": "2025-03-30"},
        "2026-10-13",
    )
    assert result["status"] == "answered"
    assert [item["start_date"] for item in result["calculations"]] == ["2025-01-02"] * 2
    assert result["calculations"][0]["growth_percent"] == "20.0"
    assert result["calculations"][1]["relative_delta_pp"] == "-10.0"
    assert "Delta vs AAPL" in result["answer"]
    assert session.call_tool.await_count == 4
    assert peak == 1


async def test_no_common_dates_does_not_compare():
    session = AsyncMock()

    async def fetch(name, arguments):
        day = "2025-01-02" if arguments["symbol"] == "AAPL" else "2025-01-03"
        return response(f"datetime;close\n{day};100")

    session.call_tool.side_effect = fetch
    result = await calculate_price_growth(
        session,
        ["AAPL", "MSFT"],
        {"start_date": "2025-01-05", "end_date": "2025-03-30"},
        "2026-10-13",
    )
    assert result["status"] == "error"
    assert "No common observation dates" in result["answer"]


async def test_comparison_today_uses_previous_day_close():
    session = AsyncMock()

    async def fetch(name, arguments):
        assert name == "get_time_series"
        return response(f"datetime;close\n{arguments['end_date']};100")

    session.call_tool.side_effect = fetch
    result = await calculate_price_growth(
        session,
        ["AAPL", "MSFT"],
        {"start_date": "2026-04-13", "end_date": "2026-10-13"},
        "2026-10-13",
    )
    assert result["status"] == "answered"
    assert all(item["end_date"] == "2026-10-12" for item in result["calculations"])


async def test_independent_history_and_quote_run_concurrently():
    import asyncio

    session = AsyncMock()
    started = set()
    ready = asyncio.Event()

    async def fetch(name, arguments):
        started.add(name)
        if len(started) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 1)
        if name == "get_quote":
            return response("datetime;close\n2026-10-13;110")
        return response("datetime;close\n2026-04-13;100")

    session.call_tool.side_effect = fetch
    result = await calculate_price_growth(
        session, "AAPL", {"start_date": "2026-04-13", "end_date": "2026-10-13"}, "2026-10-13"
    )
    assert result["status"] == "answered"
    assert started == {"get_time_series", "get_quote"}


async def test_more_than_four_assets_requires_clarification_without_calls():
    session = AsyncMock()
    result = await calculate_price_growth(
        session,
        ["AAPL", "MSFT", "META", "NVDA", "TSLA"],
        {"start_date": "2025-01-01", "end_date": "2025-02-01"},
        "2026-10-13",
    )
    assert result["status"] == "clarify"
    assert "four assets" in result["answer"]
    session.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "body",
    [
        '{"status":"error","code":429,"message":"Too many requests"}',
        "Error: You have run out of API credits for the current minute. "
        "9 API credits were used, with the current limit being 8.",
    ],
)
async def test_growth_rate_limit_suggests_one_minute(body):
    session = AsyncMock()
    session.call_tool.return_value = response(body)
    result = await calculate_price_growth(
        session, "AAPL", {"start_date": "2025-01-01", "end_date": "2025-02-01"}, "2026-10-13"
    )
    assert result["status"] == "error"
    assert "try again after one minute" in result["answer"]
    assert session.call_tool.await_count == 1
