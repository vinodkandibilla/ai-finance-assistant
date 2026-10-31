import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from finassist.web_app.components.portfolio import allocation_chart, comparison_rows
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.portfolio import sample_snapshot, snapshots

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


def test_comparison_table_preserves_missing_performance() -> None:
    first = sample_snapshot()
    second = first.model_copy(update={"customer_id": "OTHER", "performance_complete": True,
                                       "unrealized_gain_loss": -250.0})
    rows = comparison_rows([first, second])
    assert rows[0]["Largest holding"] == "60.0%"
    assert rows[0]["Holdings"] == "3"
    assert rows[0]["Unrealized gain/loss (USD)"] == "Unavailable"
    assert rows[1]["Unrealized gain/loss (USD)"] == "$-250.00"


def payload(customer: str = "C1") -> dict:
    result = sample_snapshot().model_dump()
    result.update(
        customer_id=customer,
        manager_id="M1",
        data_source="synthetic_test_data",
        price_basis="frozen_historical_close_usd",
    )
    for holding in result["holdings"]:
        holding["as_of_date"] = "2026-10-09"
    return result


def test_calculation_and_comparison() -> None:
    response = QueryResponse(
        intent="portfolio",
        answer="",
        calculation=payload(),
        calculations=[{"status": "success", "portfolios": [payload("C2")]}],
    )
    values = snapshots(response)
    assert [p.customer_id for p in values] == ["C1", "C2"]
    chart = allocation_chart(values[0], "asset_type")
    assert list(chart.data[0].values) == [60000, 30000, 10000]
    assert list(chart.data[0].labels) == ["Stocks", "Bonds", "Cash"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"portfolio_value": 1},
        {"portfolio_value": float("nan")},
        {"holdings": []},
        {"status": "error"},
        {"data_source": ""},
        {"price_basis": ""},
        {"holdings": [{"symbol": "X", "market_value": 100000, "holding_allocation_pct": 20}]},
        {"performance_complete": True, "cost_basis": 90000, "unrealized_gain_loss": 50000},
    ],
)
def test_inadequate_payloads_do_not_chart(mutation: dict) -> None:
    data = payload()
    data.update(mutation)
    # NaN is deliberately injected into the wire-model dictionary to exercise snapshot validation.
    response = QueryResponse(intent="portfolio", answer="", calculation=data)
    assert snapshots(response) == []


@pytest.mark.parametrize("status", ["error", "clarification"])
def test_response_failures_do_not_chart(status: str) -> None:
    assert (
        snapshots(
            QueryResponse(intent="portfolio", answer="", status=status, calculation=payload())
        )
        == []
    )


def test_api_dashboard_and_missing_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("No automatic API calls"))
    app = AppTest.from_file(str(APP), default_timeout=10)
    response = QueryResponse(intent="portfolio", answer="Snapshot", calculation=payload())
    app.session_state["portfolio_landing_response"] = response
    app.run().switch_page("views/portfolio.py").run()
    assert not app.exception
    assert any(m.value == "$100,000.00" for m in app.metric)
    assert len(app.get("plotly_chart")) == 3
    assert any("Performance is not shown" in item.value for item in app.info)
    assert any("2026-10-09" in item.value for item in app.caption)


def test_empty_and_explicit_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("No API calls"))
    app = (
        AppTest.from_file(str(APP), default_timeout=10)
        .run()
        .switch_page("views/portfolio.py")
        .run()
    )
    assert not app.exception
    assert not app.metric
    assert any("No adequate" in item.value for item in app.info)
    assert not any(radio.label == "Dashboard data" for radio in app.radio)


def test_optional_scope_fields_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def send(client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, request=request, json={"intent": "portfolio", "answer": "Result"}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    app = (
        AppTest.from_file(str(APP), default_timeout=10)
        .run()
        .switch_page("views/portfolio.py")
        .run()
    )
    assert not any(e.label == "Optional portfolio request scope" for e in app.expander)
    app.chat_input[0].set_value("Analyze this portfolio").run()
    assert not app.exception
    assert requests == [
        (
            "/query",
            {
                "query": "Analyze this portfolio",
            },
        )
    ]


def test_comparison_selection_and_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("No API calls"))
    first = payload()
    first.update(
        performance_complete=True,
        cost_basis=80000,
        unrealized_gain_loss=20000,
        unrealized_gain_loss_pct=25,
    )
    response = QueryResponse(
        intent="portfolio",
        answer="Comparison",
        calculations=[{"status": "success", "portfolios": [first, payload("C2")]}],
    )
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["portfolio_landing_response"] = response
    app.run().switch_page("views/portfolio.py").run()
    assert not app.exception
    assert any(m.value == "$20,000.00" for m in app.metric)
    assert len(app.get("plotly_chart")) == 3
    app.button(key="portfolio_customer_C2").click().run()
    assert not any(b.key == "portfolio_customer_C2" for b in app.button)
    assert app.button(key="portfolio_customer_C1").label == "C1"
    assert not app.exception
    assert not any(m.value == "$20,000.00" for m in app.metric)
    assert any("Performance is not shown" in item.value for item in app.info)


def test_unrecognized_provenance_is_not_used() -> None:
    data = payload()
    data["data_source"] = "live"
    assert snapshots(QueryResponse(intent="portfolio", answer="", calculation=data)) == []


@pytest.mark.parametrize("manager", [True, False])
def test_landing_loads_identity_scoped_portfolios_without_refresh_button(monkeypatch, manager):
    calls = []
    first = {**payload("C1"), "customer_name": "Demo customer one"}
    second = {**payload("C2"), "customer_name": "Demo customer two"}
    response = QueryResponse(
        intent="portfolio",
        answer="Loaded",
        calculation=(
            {"status": "success", "manager_id": "PM001", "portfolios": [first, second]}
            if manager
            else first
        ),
    )

    def fetch():
        calls.append(True)
        return response

    monkeypatch.setattr("finassist.web_app.components.portfolio.fetch_current_portfolios", fetch)
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.switch_page("views/portfolio.py").run()
    assert not app.exception
    assert len(calls) == 1
    assert any(
        s.value == ("Your customers" if manager else "Your portfolio") for s in app.subheader
    )
    if manager:
        assert not any(b.key == "portfolio_customer_C1" for b in app.button)
        assert app.button(key="portfolio_customer_C2").label == "C2"
        assert app.session_state["portfolio_selected_customer"] == "C1"
    assert not any(b.key == "refresh_portfolio_landing" for b in app.button)
    app.run()
    assert len(calls) == 1
    assert not app.exception


@pytest.mark.parametrize("manager_id", [None, "PM002"])
def test_portfolio_landing_request_uses_server_identity_only(monkeypatch, manager_id):
    from finassist.web_app.services.portfolio import fetch_current_portfolios

    requests = []

    def send(client, request, **kwargs):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200, request=request, json={"intent": "portfolio", "answer": "Loaded"}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    fetch_current_portfolios(manager_id)
    assert requests[0][0] == "/query"
    assert set(requests[0][1]) == {"query"}
    query = requests[0][1]["query"]
    if manager_id:
        assert "all customer portfolios assigned" in query
        assert "For my customer account" not in query
    else:
        assert "available to my current account" in query
    assert "all concentration levels" in query
    assert "unavailable portfolios" in query
    assert "trusted application identity" not in query
    assert "Use portfolio tools" not in query
    assert "Do not ask" not in query


def test_customer_directory_search_and_detail_tabs(monkeypatch):
    response = QueryResponse(
        intent="portfolio",
        answer="Loaded",
        calculation={
            "status": "success",
            "manager_id": "PM001",
            "portfolios": [
                {**payload("C1"), "customer_name": "Alex Demo"},
                {**payload("C2"), "customer_name": "Jordan Demo"},
            ],
        },
    )
    monkeypatch.setattr(
        "finassist.web_app.components.portfolio.fetch_current_portfolios", lambda: response
    )
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.switch_page("views/portfolio.py").run()
    assert [tab.label for tab in app.tabs] == ["Summary", "Positions", "Allocation"]
    app.text_input(key="portfolio_customer_search").input("jordan").run()
    assert not app.exception
    assert not any(button.key == "portfolio_customer_C2" for button in app.button)
    assert not any(button.key == "portfolio_customer_C1" for button in app.button)
    assert app.session_state["portfolio_selected_customer"] == "C2"
    assert any(h.value == "C2 - Jordan Demo" for h in app.subheader)
    app.text_input(key="portfolio_customer_search").input("unmatched").run()
    assert any("No customers match" in item.value for item in app.info)
    assert not app.exception


def test_performance_cards_show_losses_and_backend_concentration_flag():
    data = payload()
    data.update(
        performance_complete=True,
        cost_basis=120000,
        unrealized_gain_loss=-20000,
        unrealized_gain_loss_pct=-20000 / 120000 * 100,
        concentration={"is_concentrated": True},
    )
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["portfolio_landing_response"] = QueryResponse(
        intent="portfolio", answer="Loaded", calculation=data
    )
    app.run().switch_page("views/portfolio.py").run()
    assert not app.exception
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Unrealized gain / loss (USD)"] == "$-20,000.00"
    assert values["Unrealized gain / loss (%)"] == "-16.67%"
    assert values["Concentration"] == "Flagged"


@pytest.mark.parametrize(
    "gains,expected",
    [
        ([10, 20, 0], "Gains"),
        ([-10, -20, 0], "Losses"),
        ([10, -20, 0], "Mixed"),
        ([0, 0, 0], "Unchanged"),
        ([10, None, 0], "Unavailable"),
    ],
)
def test_performance_label_uses_holdings_not_scenario_text(gains, expected):
    from finassist.web_app.services.portfolio import performance_label

    snapshot = sample_snapshot()
    snapshot.performance_complete = True
    for holding, gain in zip(snapshot.holdings, gains, strict=True):
        holding.unrealized_gain_loss = gain
    assert performance_label(snapshot) == expected


def test_manager_selector_reloads_customers_and_scopes_chat(monkeypatch):
    from finassist.web_app.services.models import PortfolioManagerDirectory

    monkeypatch.setattr(
        "finassist.web_app.components.portfolio.fetch_portfolio_managers",
        lambda: PortfolioManagerDirectory.model_validate(
            {
                "managers": [
                    {"manager_id": "PM001", "manager_name": "Manager One"},
                    {"manager_id": "PM002", "manager_name": "Manager Two"},
                ],
                "current_manager_id": "PM001",
                "can_switch": True,
            }
        ),
    )
    calls = []

    def fetch(manager_id):
        calls.append(manager_id)
        row = payload("C001" if manager_id == "PM001" else "C006")
        row["manager_id"] = manager_id
        return QueryResponse(
            intent="portfolio",
            answer="Available",
            calculation={
                "status": "success",
                "portfolios": [row],
            },
        )

    monkeypatch.setattr("finassist.web_app.components.portfolio.fetch_current_portfolios", fetch)
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={"intent": "portfolio", "answer": "Done"})

    monkeypatch.setattr(httpx.Client, "send", send)
    app = AppTest.from_file(str(APP)).run().switch_page("views/portfolio.py").run()
    assert calls == ["PM001"]
    assert app.session_state["portfolio_selected_customer"] == "C001"
    app.selectbox(key="portfolio_manager_id").select("PM002").run()
    assert calls == ["PM001", "PM002"]
    assert app.session_state["portfolio_selected_customer"] == "C006"
    app.chat_input[0].set_value("Analyze my customers").run()
    assert requests[-1].headers["X-Portfolio-Manager"] == "PM002"
    assert not app.exception
