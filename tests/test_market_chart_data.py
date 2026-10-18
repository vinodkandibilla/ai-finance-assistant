"""Provider observations reach chart payloads without parsing generated answers."""

import json
from types import SimpleNamespace

from finassist.agents.market_analysis.chart_data import chart_observation


def test_quote_and_weekly_provider_payloads() -> None:
    for payload in [
        {"symbol": "AAPL", "exchange": "NASDAQ", "currency": "USD", "close": "201"},
        {
            "meta": {"symbol": "AAPL", "exchange": "NASDAQ", "currency": "USD"},
            "values": [
                {
                    "datetime": "2026-10-01",
                    "open": "200",
                    "high": "202",
                    "low": "199",
                    "close": "201",
                }
            ],
        },
    ]:
        result = SimpleNamespace(
            isError=False,
            structuredContent=None,
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        )
        chart = chart_observation(result, tool="get_time_series", symbol="AAPL", interval="1week")
        assert chart and chart["symbol"] == "AAPL" and chart["currency"] == "USD"
        assert chart["values"][0]["close"] == "201"
        assert chart["interval"] == "1week"


def test_provider_error_is_not_chart_data() -> None:
    result = SimpleNamespace(
        isError=False, structuredContent={"status": "error", "code": 429}, content=[]
    )
    assert chart_observation(result, tool="get_quote", symbol="AAPL", interval="quote") is None


def test_wrapped_provider_csv_is_exposed() -> None:
    csv_text = (
        "symbol;exchange;currency;datetime;open;high;low;close\n"
        "AAPL;NASDAQ;USD;2020-01-02;100;110;95;105"
    )
    result = SimpleNamespace(isError=False, structuredContent={"result": csv_text}, content=[])
    chart = chart_observation(result, tool="get_quote", symbol="AAPL", interval="quote")
    assert chart and chart["exchange"] == "NASDAQ"
    assert chart["values"][0]["close"] == "105"
