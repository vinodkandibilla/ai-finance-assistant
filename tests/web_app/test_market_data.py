import json

import httpx
import pytest

from finassist.web_app.services.market import fetch_market_snapshot, market_chart_data
from finassist.web_app.services.models import QueryResponse


@pytest.mark.parametrize(
    "interval,window,expected_interval",
    [("Daily", "1 month", "1day"), ("Weekly", "3 months", "1week")],
)
def test_market_fetch_uses_only_query(monkeypatch, interval, window, expected_interval) -> None:
    calls = []

    def send(client, request, **kwargs):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, request=request, json={"intent": "market", "answer": "Available"}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    fetch_market_snapshot("AAPL", interval)
    assert calls[0][0] == "/query"
    assert set(calls[0][1]) == {"query"}
    assert f"{interval.lower()} OHLC" in calls[0][1]["query"]
    assert f"last {window}" in calls[0][1]["query"]
    assert "under 120 words" in calls[0][1]["query"]
    assert "Do not reproduce historical rows" in calls[0][1]["query"]
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
