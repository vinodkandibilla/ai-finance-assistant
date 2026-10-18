from datetime import date

import pytest

from finassist.agents.market_analysis.timeframe import resolve_timeframe

TODAY = date(2026, 10, 13)


@pytest.mark.parametrize(
    "query, interval, size",
    [
        ("QQQ weekly data for the last 12 weeks", "1week", 12),
        ("AAPL last 5 days", "1day", 5),
        ("AAPL monthly history for the past 6 months", "1month", 6),
    ],
)
def test_relative_bar_counts(query, interval, size):
    result = resolve_timeframe(query, TODAY)
    assert result["status"] == "ready"
    assert result["parameters"] == {"interval": interval, "outputsize": size}
    assert result["as_of"] == "2026-10-13"


@pytest.mark.parametrize(
    "query, start, end",
    [
        ("AAPL daily data for last 12 weeks", "2026-07-21", "2026-10-13"),
        ("AAPL fiscal Q2 data from 2025-01-01 to 2025-03-29", "2025-01-01", "2025-03-29"),
    ],
)
def test_date_boundaries(query, start, end):
    result = resolve_timeframe(query, TODAY)
    assert result["parameters"] == {
        "interval": "1day",
        "start_date": start,
        "end_date": end,
        "outputsize": 100,
    }


def test_month_subtraction_clamps_day_in_leap_year():
    result = resolve_timeframe("daily data last 1 month", date(2024, 3, 31))
    assert result["parameters"]["start_date"] == "2024-02-29"


@pytest.mark.parametrize(
    "query",
    [
        "AAPL fiscal Q2 2025",
        "QQQ last quarter",
        "QQQ calendar Q2 2025",
        "AAPL Q4 2025",
        "AAPL Q2 results 2025",
        "from last quarter results",
        "AAPL Q2",
        "last 0 weeks",
        "last 101 weeks",
        "QQQ historical prices",
        "from 2025-02-30 to 2025-03-01",
        "from 2025-03-01 to 2025-02-01",
        "from 2025-01-01 to 2027-01-01",
        "from 2025-01-01",
    ],
)
def test_uncertain_or_invalid_windows_query_clarification(query):
    result = resolve_timeframe(query, TODAY)
    assert result["status"] == "clarify"
    assert result["clarification_question"]
    assert "parameters" not in result


def test_hundred_records_is_allowed():
    assert resolve_timeframe("last 100 weeks", TODAY)["parameters"]["outputsize"] == 100


def test_large_daily_window_requires_narrowing():
    assert resolve_timeframe("daily from 2025-01-01 to 2025-12-31", TODAY)["status"] == "clarify"
    assert resolve_timeframe("weekly from 2025-01-01 to 2025-12-31", TODAY)["status"] == "ready"


@pytest.mark.parametrize(
    "window",
    ["last month", "last one month", "over the last month", "past one month", "last the month"],
)
def test_natural_language_month_price_change_has_date_bounds(window):
    result = resolve_timeframe(f"How much has NVDA’s price changed {window}?", TODAY)
    assert result["status"] == "ready"
    assert result["parameters"] == {
        "interval": "1day",
        "start_date": "2026-09-13",
        "end_date": "2026-10-13",
        "outputsize": 100,
    }


def test_spelled_window_in_clarification_retains_original_growth_request():
    query = "How much has NVDA’s price changed?\nAdditional details: last one month"
    result = resolve_timeframe(query, date(2024, 3, 31))
    assert result["status"] == "ready"
    assert result["parameters"]["start_date"] == "2024-02-29"
    assert (
        resolve_timeframe("NVDA daily history for the last two months", TODAY)["status"] == "ready"
    )
