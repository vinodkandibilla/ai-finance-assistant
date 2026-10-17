import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from finassist.web_app.components.chat import ChatState, Message
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.showcase import market_endpoints, project_goal, safe_source

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


def click(app: AppTest, label: str) -> None:
    next(b for b in app.button if b.label == label).click().run()


def app_for(page: str) -> AppTest:
    return (
        AppTest.from_file(str(APP), default_timeout=10).run().switch_page(f"views/{page}.py").run()
    )


def test_goal_math() -> None:
    result = project_goal(1000, 100, 50, 0, 12)
    assert result.balances[-1] == 700
    assert result.required_monthly == 75
    assert result.shortfall == 300
    assert project_goal(1000, 1000, 100, 12, 1).balances[-1] == 1110
    assert project_goal(100, 200, 0, 0, 12).required_monthly == 0
    compound = project_goal(10000, 1000, 100, 6, 24)
    expected = 1000 * 1.005**24 + 100 * (1.005**24 - 1) / 0.005
    assert compound.balances[-1] == round(expected, 2)
    assert len(compound.balances) == 25


@pytest.mark.parametrize(
    "inputs", [(-1, 0, 1, 1, 12), (1, float("nan"), 1, 1, 12), (1, 1, 1, 101, 12), (1, 1, 1, 0, 0)]
)
def test_invalid_goal_inputs(inputs: tuple) -> None:
    with pytest.raises(ValueError):
        project_goal(*inputs)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://news.test/article", "news.test"),
        ("javascript:alert(1)", None),
        ("https://u:p@news.test", None),
        ("not a link", None),
    ],
)
def test_safe_links(url: str, expected: str | None) -> None:
    assert safe_source(url) == expected


def test_structured_market_endpoints() -> None:
    result = market_endpoints(
        QueryResponse(
            intent="market",
            answer="",
            calculation={
                "symbol": "AAPL",
                "currency": "USD",
                "start_date": "2026-09-01",
                "end_date": "2026-10-01",
                "start_price": "100",
                "end_price": "110",
            },
        )
    )
    assert len(result) == 1
    assert result[0].end_price == 110
    assert market_endpoints(QueryResponse(intent="market", answer="Price is 110")) == []
    assert (
        market_endpoints(QueryResponse(intent="market", answer="", calculation={"start_price": -1}))
        == []
    )


def test_watchlist_is_local_and_sample_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **kw: pytest.fail("No API calls"))
    app = app_for("market")
    next(t for t in app.text_input if t.label == "Watchlist symbol").input("aapl")
    click(app, "Add to watchlist")
    assert next(s for s in app.selectbox if s.label.startswith("Saved symbols")).value == "AAPL"
    click(app, "Add to watchlist")
    assert len(next(s for s in app.selectbox if s.label.startswith("Saved symbols")).options) == 1
    assert not app.get("plotly_chart")
    assert any("No validated OHLC" in item.value for item in app.info)
    click(app, "Remove selected symbol")
    assert any("watchlist is empty" in i.value for i in app.info)
    assert not app.exception


@pytest.mark.parametrize(
    "page,action",
    [
        ("goals", "Ask AI about this scenario"),
        ("news", "Get news summary"),
    ],
)
def test_showcase_requests_use_query(
    page: str, action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def send(client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, request=request, json={"intent": page, "answer": "API result"})

    monkeypatch.setattr(httpx.Client, "send", send)
    app = app_for(page)
    if page == "market":
        next(t for t in app.text_input if t.label == "Stock symbol or company").input("Apple")
    elif page == "news":
        next(t for t in app.text_input if t.label == "News topic").input("Banking")
    click(app, action)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0][0] == "/query"
    assert set(calls[0][1]) == {"query"}
    assert len(app.chat_message) == 2
    app.run()
    assert len(calls) == 1


def test_goal_inputs_update_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **kw: pytest.fail("No API calls"))
    app = app_for("goals")
    first = app.metric[0].value
    next(n for n in app.number_input if n.label == "Monthly contribution (USD)").set_value(
        400.0
    ).run()
    assert app.metric[0].value != first
    assert any("LOCAL HYPOTHETICAL" in i.value for i in app.info)
    assert not app.exception


def test_news_source_metadata_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **kw: pytest.fail("No API calls"))
    app = AppTest.from_file(str(APP), default_timeout=10)
    response = QueryResponse(
        intent="news",
        answer="A returned summary",
        citations=["https://news.test/article", "javascript:bad()"],
    )
    app.session_state["conversation"] = ChatState(
        messages=[Message("assistant", response.answer, response)]
    )
    app.run().switch_page("views/news.py").run()
    assert not app.exception
    assert len(app.get("link_button")) == 2
    assert any("Publication dates are not structured" in c.value for c in app.caption)
    assert any("javascript:bad()" in t.value for t in app.text)


@pytest.mark.parametrize("action", ["Ask for latest available quote", "chat"])
def test_market_requests_are_scoped_to_us_nasdaq(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    calls = []

    def send(client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, request=request, json={"intent": "market", "answer": "AAPL on NASDAQ"}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    app = app_for("market")
    if action == "Search stocks":
        next(t for t in app.text_input if t.label == "Stock symbol or company").input("AAPL")
        click(app, action)
    elif action == "Ask for latest available quote":
        next(t for t in app.text_input if t.label == "Watchlist symbol").input("AAPL")
        click(app, "Add to watchlist")
        click(app, action)
    else:
        app.chat_input[0].set_value("AAPL quote").run()
    assert not app.exception
    assert len(calls) == 1
    assert set(calls[0]) == {"query"}
    assert "United States, NASDAQ-listed equities only, currency USD" in calls[0]["query"]
    assert "Exclude all other exchanges" in calls[0]["query"]
    assert "under 120 words" in calls[0]["query"]


def test_watchlist_selects_each_new_symbol_and_preserves_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **kw: pytest.fail("No API calls"))
    app = app_for("market")
    for ticker in ["AAPL", "msft", "NVDA"]:
        next(t for t in app.text_input if t.label == "Watchlist symbol").input(ticker)
        click(app, "Add to watchlist")
        assert not app.exception
        assert app.selectbox(key="market_selected_symbol").value == ticker.upper()
        assert any(f"Added {ticker.upper()}" in item.value for item in app.success)
    assert app.session_state["market_watchlist"] == ["AAPL", "MSFT", "NVDA"]
    assert app.selectbox(key="market_selected_symbol").options == ["AAPL", "MSFT", "NVDA"]
    assert any("Saved symbols (3): AAPL, MSFT, NVDA" in item.value for item in app.caption)
    click(app, "Remove selected symbol")
    assert app.session_state["market_watchlist"] == ["AAPL", "MSFT"]
    assert not app.exception


@pytest.mark.parametrize("status", [None, "error"])
def test_market_fetches_each_rerun_without_ui_cache(
    monkeypatch: pytest.MonkeyPatch, status: str | None
) -> None:
    requested = []

    def fetch(symbol: str, interval: str) -> QueryResponse:
        requested.append((symbol, interval))
        return QueryResponse(intent="market", answer="Data unavailable", status=status)

    monkeypatch.setattr("finassist.web_app.components.showcase.fetch_market_snapshot", fetch)
    app = app_for("market")
    assert requested == [("AAPL", "Daily")]
    assert next(r for r in app.radio if r.label == "Interval").value == "Daily"
    assert app.text_input(key="market_chart_symbol").value == "AAPL"
    next(r for r in app.radio if r.label == "Interval").set_value("Weekly").run()
    assert requested[-1] == ("AAPL", "Weekly")
    app.run()
    assert requested == [("AAPL", "Daily"), ("AAPL", "Weekly"), ("AAPL", "Weekly")]
    app.text_input(key="market_chart_symbol").input("MSFT").run()
    assert requested[-1] == ("MSFT", "Weekly")
    app.run()
    assert requested.count(("MSFT", "Weekly")) == 2
    assert not app.exception
    assert not app.get("plotly_chart")


@pytest.mark.parametrize("intent", ["news", "tax", "market", "portfolio", "finance_qa", "goals"])
def test_market_content_never_refreshes_from_chat_intent(intent) -> None:
    from finassist.web_app.components.chat import ChatState, Message, _refresh_matching_content

    state = ChatState(
        messages=[Message("assistant", "Answer", QueryResponse(intent=intent, answer="Answer"))]
    )
    assert not _refresh_matching_content(state, "market")


def test_comparison_missing_data_and_duplicate_symbols(monkeypatch):
    requested = []

    def fetch(symbol, interval):
        requested.append((symbol, interval))
        return QueryResponse(intent="market", answer="No data")

    monkeypatch.setattr("finassist.web_app.components.showcase.fetch_market_snapshot", fetch)
    app = app_for("market")
    app.radio(key="market_view").set_value("Compare stocks").run()
    assert requested[-2:] == [("AAPL", "Daily"), ("MSFT", "Daily")]
    assert any("No validated" in element.value for element in app.info)
    app.text_input(key="compare_second").input("AAPL").run()
    assert any("different symbols" in element.value for element in app.info)
    assert not app.exception


@pytest.mark.parametrize("interval,provider_interval", [("Daily", "1day"), ("Weekly", "1week")])
def test_comparison_chart_uses_validated_histories(monkeypatch, interval, provider_interval):
    def fetch(symbol, requested_interval):
        prices = [100, 110] if symbol == "AAPL" else [200, 180]
        return QueryResponse(
            intent="market",
            answer="Available",
            calculations=[
                {
                    "type": "market_observations",
                    "symbol": symbol,
                    "currency": "USD",
                    "exchange": "NASDAQ",
                    "interval": provider_interval,
                    "values": [
                        {
                            "datetime": day,
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                        }
                        for day, price in zip(["2026-01-02", "2026-01-09"], prices, strict=True)
                    ],
                }
            ],
        )

    monkeypatch.setattr("finassist.web_app.components.showcase.fetch_market_snapshot", fetch)
    app = app_for("market")
    app.radio(key="market_view").set_value("Compare stocks").run()
    app.radio(key="compare_interval").set_value(interval).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    chart = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert chart["data"][0]["y"] == [100.0, 110.0]
    assert chart["data"][1]["y"] == [200.0, 180.0]
    assert chart["layout"]["yaxis"]["title"]["text"] == "Closing price (USD)"
    assert chart["data"][1]["customdata"] == [[200.0, 100.0], [180.0, 90.0]]
    assert [metric.value for metric in app.metric] == ["+10.00%", "-10.00%"]
