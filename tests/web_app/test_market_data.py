import json

import httpx
import pytest

from finassist.web_app.services.market import fetch_market_snapshot, market_chart_data
from finassist.web_app.services.models import QueryResponse


@pytest.mark.parametrize(
    "interval,window,expected_interval",
    [("Daily", "1 month", "1day"), ("Weekly", "3 months", "1week")],
)
@pytest.mark.parametrize("symbol", ["AAPL", ["AAPL", "MSFT"]])
def test_market_fetch_uses_only_query(
    monkeypatch, interval, window, expected_interval, symbol
) -> None:
    calls = []

    def send(client, request, **kwargs):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, request=request, json={"intent": "market", "answer": "Available"}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    fetch_market_snapshot(symbol, interval)
    assert calls[0][0] == "/query"
    assert set(calls[0][1]) == {"query", "market_scope"}
    assert calls[0][1]["market_scope"] == "us_nasdaq_usd"
    assert f"{interval.lower()} OHLC" in calls[0][1]["query"]
    assert f"last {window}" in calls[0][1]["query"]
    target = symbol if isinstance(symbol, str) else "AAPL and MSFT"
    assert len(calls) == 1
    if isinstance(symbol, list):
        assert f"For each of {target}, show latest quote" in calls[0][1]["query"]
    else:
        assert f"Show {target}'s latest quote" in calls[0][1]["query"]
    assert "Scope:" not in calls[0][1]["query"]
    assert len(calls[0][1]["query"].split()) < 65
    from datetime import date

    from finassist.agents.market_analysis.timeframe import resolve_timeframe

    assert (
        resolve_timeframe(calls[0][1]["query"], date(2026, 10, 15))["parameters"]["interval"]
        == expected_interval
    )


def test_chart_accepts_only_matching_valid_provider_bars() -> None:
    data = {
        "type": "market_observations",
        "symbol": "AAPL",
        "currency": "USD",
        "exchange": "NASDAQ",
        "interval": "1week",
        "values": [
            {"datetime": "2026-10-01", "open": "200", "high": "202", "low": "199", "close": "201"},
            {"datetime": "2026-10-02", "open": "200", "high": "190", "low": "199", "close": "201"},
        ],
    }
    response = QueryResponse(intent="market", answer="", calculations=[data])
    bars, _ = market_chart_data(response, "AAPL", "Weekly")
    assert len(bars) == 1 and bars[0].close == 201
    assert not market_chart_data(response, "MSFT", "Weekly")[0]
    assert not market_chart_data(response, "AAPL", "Daily")[0]
    response.status = "error"
    assert not market_chart_data(response, "AAPL", "Weekly")[0]


def test_dashboard_draws_provider_ohlc(monkeypatch) -> None:
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    response = QueryResponse(
        intent="market",
        answer="Available",
        calculations=[
            {
                "type": "market_observations",
                "symbol": "AAPL",
                "exchange": "NASDAQ",
                "currency": "USD",
                "interval": "1day",
                "values": [
                    {
                        "datetime": "2020-01-02",
                        "open": "100",
                        "high": "110",
                        "low": "95",
                        "close": "105",
                        "volume": "1000",
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        "finassist.web_app.components.showcase.fetch_market_snapshot",
        lambda symbol, interval: response,
    )
    root = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(root / "src/finassist/web_app/app.py")).run()
    app.switch_page("views/market.py").run()
    assert not app.exception
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert figure["data"][0]["type"] == "candlestick"
    assert figure["data"][0]["close"] == [105.0]
    assert figure["data"][0]["high"] == [110.0]


def test_comparison_aligns_shared_dates_and_rebases_prices():
    from finassist.web_app.services.market import MarketBar, comparison_series

    def bar(day, price):
        return MarketBar(datetime=day, open=price, high=price, low=price, close=price)

    first = [bar("2026-01-01", 10), bar("2026-01-02", 20), bar("2026-01-03", 30)]
    second = [bar("2026-01-02", 100), bar("2026-01-03", 90), bar("2026-01-04", 110)]
    assert comparison_series(first, second) == (["2026-01-02", "2026-01-03"], [100, 150], [100, 90])
    assert comparison_series(first[:1], second) == ([], [], [])


@pytest.mark.parametrize("symbols,expected", [(["INTC"], "INTC"), (["INTC", "AAPL"], None)])
def test_resolved_symbol_requires_one_structured_listing(symbols, expected):
    from finassist.web_app.services.market import resolved_market_symbol

    response = QueryResponse(intent="market", answer="INTEL resolved to INTC", calculations=[
        {"type": "market_observations", "symbol": symbol, "currency": "USD",
         "exchange": "NASDAQ", "values": []}
        for symbol in symbols
    ])
    assert resolved_market_symbol(response) == expected
    assert resolved_market_symbol(QueryResponse(intent="market", answer="INTC")) is None
