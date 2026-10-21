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
    if label == "Add to watchlist":
        app.run()
    else:
        next(b for b in app.button if b.label == label).click().run()


def symbol_input(app, suffix=""):
    key = f"watchlist_symbol_input{suffix}"
    if not any(widget.key == key for widget in app.text_input):
        app.button(key=f"watchlist_show_input{suffix}").click().run()
    return app.text_input(key=key)


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
    symbol_input(app).input("aapl")
    click(app, "Add to watchlist")
    assert app.button(key="watchlist_quote_AAPL").label == "AAPL"
    click(app, "Add to watchlist")
    assert app.session_state["market_watchlist"] == ["AAPL"]
    assert not app.get("plotly_chart")
    assert any("No validated OHLC" in item.value for item in app.info)
    app.button(key="watchlist_remove_AAPL").click().run()
    assert any("watchlist is empty" in i.value for i in app.info)
    assert not app.exception


@pytest.mark.parametrize(
    "page,action",
    [
        ("goals", "Ask AI about this scenario"),
        ("news", "Search news"),
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
        next(t for t in app.text_input if t.label == "Search topic").input("Banking")
    click(app, action)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0][0] == "/query"
    assert set(calls[0][1]) == {"query"}
    assert len(app.chat_message) == (0 if page == "news" else 2)
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
    assert any("Choose a topic and period" in item.value for item in app.info)
    assert not any(button.label == "Read article" for button in app.get("link_button"))


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
        symbol_input(app).input("AAPL")
        click(app, "Add to watchlist")
        app.button(key="watchlist_quote_AAPL").click().run()
    else:
        app.chat_input[0].set_value("AAPL quote").run()
    assert not app.exception
    assert len(calls) == 1
    if action == "Ask for latest available quote":
        assert calls[0]["market_scope"] == "us_nasdaq_usd"
        assert "Scope:" not in calls[0]["query"]
    else:
        assert set(calls[0]) == {"query"}
        assert "United States, NASDAQ-listed equities only, currency USD" in calls[0]["query"]
        assert "No depositary receipts" in calls[0]["query"]
    assert "under 120 words" in calls[0]["query"]


def test_watchlist_selects_each_new_symbol_and_preserves_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **kw: pytest.fail("No API calls"))
    app = app_for("market")
    for ticker in ["AAPL", "msft", "NVDA"]:
        symbol_input(app).input(ticker)
        click(app, "Add to watchlist")
        assert not app.exception
        assert app.button(key=f"watchlist_quote_{ticker.upper()}").label == ticker.upper()
        assert any(f"Added {ticker.upper()}" in item.value for item in app.caption)
    assert app.session_state["market_watchlist"] == ["AAPL", "MSFT", "NVDA"]
    assert all(app.button(key=f"watchlist_quote_{symbol}") for symbol in ["AAPL", "MSFT", "NVDA"])
    assert any(item.value == "Watchlist" for item in app.subheader)
    app.button(key="watchlist_remove_NVDA").click().run()
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
    assert requested == [("AAPL", "Daily"), (["AAPL", "MSFT"], "Daily")]
    assert any("No validated" in element.value for element in app.info)
    app.text_input(key="compare_second").input("AAPL").run()
    assert any("different symbols" in element.value for element in app.info)
    assert not app.exception


@pytest.mark.parametrize("interval,provider_interval", [("Daily", "1day"), ("Weekly", "1week")])
def test_comparison_chart_uses_validated_histories(monkeypatch, interval, provider_interval):
    def fetch(symbol, requested_interval):
        symbols = [symbol] if isinstance(symbol, str) else symbol
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
                        for day, price in zip(
                            ["2026-01-02", "2026-01-09"],
                            [100, 110] if symbol == "AAPL" else [200, 180],
                            strict=True,
                        )
                    ],
                }
                for symbol in symbols
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


def test_comparison_keeps_watchlist_quote_links():
    app = app_for("market")
    symbol_input(app).input("MSFT")
    click(app, "Add to watchlist")
    app.radio(key="market_view").set_value("Compare stocks").run()
    assert app.button(key="watchlist_quote_MSFT").label == "MSFT"
    assert not app.exception
    assert app.radio(key="market_view").value == "Compare stocks"
    assert not any(button.label == "Quote" for button in app.button)
    assert app.session_state["market_watchlist"] == ["MSFT"]


def test_watchlist_quote_card_is_inline_and_refreshes_on_click(monkeypatch):
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return QueryResponse(
            intent="market",
            answer="Quote",
            calculations=[
                {
                    "type": "market_observations",
                    "symbol": symbol,
                    "currency": "USD",
                    "exchange": "NASDAQ",
                    "interval": "quote",
                    "values": [{"close": "123.456", "volume": 10000, "datetime": "2026-01-02"}],
                }
            ],
        )

    monkeypatch.setattr("finassist.web_app.components.showcase.fetch_market_quote", fetch)
    app = app_for("market")
    symbol_input(app).input("AAPL")
    click(app, "Add to watchlist")
    app.button(key="watchlist_quote_AAPL").click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["$123.46", "10,000"]
    assert any("Observation: 2026-01-02" in caption.value for caption in app.caption)
    assert "conversation" not in app.session_state or not app.session_state["conversation"].messages
    app.button(key="watchlist_quote_AAPL").click().run()
    assert calls == ["AAPL", "AAPL"]
    app.button(key="watchlist_remove_AAPL").click().run()
    assert not app.metric


def test_named_watchlists_preserve_symbols_and_validate_names():
    app = app_for("market")
    symbol_input(app).input("AAPL")
    app.run()
    app.text_input(key="new_watchlist_name").input("Technology")
    click(app, "Create watchlist")
    cards = [card for card in app.expander if card.label in {"Technology", "My watchlist"}]
    assert [card.label for card in cards] == ["Technology", "My watchlist"]
    assert all(card.proto.expanded for card in cards)
    symbol_input(app, "_Technology").input("NVDA")
    app.run()
    assert app.session_state["market_watchlists"] == {
        "Technology": ["NVDA"],
        "My watchlist": ["AAPL"],
    }
    app.text_input(key="new_watchlist_name").input(" technology ")
    click(app, "Create watchlist")
    assert any("already exists" in warning.value for warning in app.warning)
    app.text_input(key="new_watchlist_name").input(" ")
    click(app, "Create watchlist")
    assert any("between 1 and 40" in warning.value for warning in app.warning)
    app.button(key="watchlist_remove_NVDA_Technology").click().run()
    assert app.session_state["market_watchlists"]["Technology"] == []
    assert app.session_state["market_watchlists"]["My watchlist"] == ["AAPL"]
    assert not app.exception


@pytest.mark.parametrize("active", [None, "Removed list"])
def test_watchlists_recover_missing_or_stale_active_selection(active):
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["market_watchlists"] = {"My list": ["AAPL"], "Tech": ["NVDA"]}
    if active is not None:
        app.session_state["market_active_watchlist"] = active
    app.run().switch_page("views/market.py").run()
    assert not app.exception
    assert app.session_state["market_active_watchlist"] == "My list"
    assert app.session_state["market_watchlists"]["Tech"] == ["NVDA"]
    app.switch_page("views/goals.py").run()
    app.switch_page("views/market.py").run()
    assert not app.exception
    assert app.session_state["market_watchlists"]["Tech"] == ["NVDA"]


def test_watchlist_quote_can_be_closed_without_removing_stock(monkeypatch):
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return QueryResponse(intent="market", answer="Latest quote is available.")

    monkeypatch.setattr("finassist.web_app.components.showcase.fetch_market_quote", fetch)
    app = app_for("market")
    symbol_input(app).input("AAPL")
    app.run()
    app.button(key="watchlist_quote_AAPL").click().run()
    app.button(key="dismiss_watchlist_quote").click().run()
    assert not app.exception
    assert app.session_state["market_watchlists"]["My watchlist"] == ["AAPL"]
    assert "watchlist_quote_result" not in app.session_state
    assert not any(b.key == "dismiss_watchlist_quote" for b in app.button)
    assert calls == ["AAPL"]


def test_watchlist_add_control_switches_between_plus_and_input():
    app = app_for("market")
    assert not any(widget.key == "watchlist_symbol_input" for widget in app.text_input)
    symbol_input(app).input("!").run()
    assert app.text_input(key="watchlist_symbol_input").value == "!"
    app.text_input(key="watchlist_symbol_input").input("AAPL").run()
    assert not any(widget.key == "watchlist_symbol_input" for widget in app.text_input)
    assert app.button(key="watchlist_show_input")
    assert app.session_state["market_watchlists"]["My watchlist"] == ["AAPL"]
    assert symbol_input(app).value == ""
    assert not app.exception

