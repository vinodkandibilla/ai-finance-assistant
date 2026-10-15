import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from finassist.web_app.components.chat import ChatState, Message
from finassist.web_app.components.portfolio import allocation_chart
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.portfolio import sample_snapshot, snapshots

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


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
    app.session_state["conversation"] = ChatState(
        messages=[Message("assistant", "Snapshot", response)]
    )
    app.run().switch_page("views/portfolio.py").run()
    assert not app.exception
    assert any(m.value == "$100,000.00" for m in app.metric)
    assert len(app.get("plotly_chart")) == 3
    assert any("Performance is not shown" in item.value for item in app.info)
    assert any("2026-10-09" in item.value for item in app.caption)
    assert any("Backend synthetic" in item.value for item in app.info)


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
    app.radio[0].set_value("Synthetic sample").run()
    assert not app.exception
    assert any("UI SAMPLE ONLY" in item.value for item in app.warning)
    assert len(app.get("plotly_chart")) == 3


def test_scope_fields_are_sent_over_query(monkeypatch: pytest.MonkeyPatch) -> None:
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
    app.selectbox(key="portfolio_scope_role").select("portfolio_manager")
    app.text_input(key="portfolio_scope_customer").input("C1")
    app.text_input(key="portfolio_scope_manager").input("M1")
    app.chat_input[0].set_value("Analyze this portfolio").run()
    assert not app.exception
    assert requests == [
        (
            "/query",
            {
                "query": "Analyze this portfolio",
                "user_role": "portfolio_manager",
                "customer_id": "C1",
                "manager_id": "M1",
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
    app.session_state["conversation"] = ChatState(
        messages=[Message("assistant", "Comparison", response)]
    )
    app.run().switch_page("views/portfolio.py").run()
    assert not app.exception
    assert any(m.value == "$20,000.00" for m in app.metric)
    assert len(app.get("plotly_chart")) == 4
    next(s for s in app.selectbox if s.label == "Customer snapshot").select("C2").run()
    assert not app.exception
    assert not any(m.value == "$20,000.00" for m in app.metric)
    assert any("Performance is not shown" in item.value for item in app.info)


def test_unrecognized_provenance_is_not_used() -> None:
    data = payload()
    data["data_source"] = "live"
    assert snapshots(QueryResponse(intent="portfolio", answer="", calculation=data)) == []
