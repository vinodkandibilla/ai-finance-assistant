import json

import httpx

from finassist.web_app.services.market import fetch_market_snapshot, market_chart_data
from finassist.web_app.services.models import QueryResponse


def test_market_fetch_uses_only_query(monkeypatch) -> None:
    calls = []

    def send(client, request, **kwargs):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, request=request, json={"intent": "market", "answer": "Available"}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    fetch_market_snapshot("AAPL", "Weekly")
    assert calls[0][0] == "/query"
    assert set(calls[0][1]) == {"query"}
    assert "weekly OHLC" in calls[0][1]["query"]
    assert "last 3 months" in calls[0][1]["query"]
    from datetime import date

    from finassist.agents.market_analysis.timeframe import resolve_timeframe

    assert (
        resolve_timeframe(calls[0][1]["query"], date(2026, 10, 15))["parameters"]["interval"]
        == "1week"
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
