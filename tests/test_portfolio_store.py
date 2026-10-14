"""Data-layer checks using actual in-memory SQLite, without provider calls."""

import sqlite3

import pytest

from ai_finance_assistant.agents.portfolio_analysis import PortfolioStore
from ai_finance_assistant.agents.portfolio_analysis.store import PRICE_AS_OF_DATE


@pytest.fixture
def store():
    with PortfolioStore() as database:
        yield database


def test_initialization_and_manager_relationships(store):
    managers = store.get_managers()
    customers = store.get_customers()
    assert store.data_source == "synthetic_test_data"
    assert len(managers) == 5
    assert len(customers) == 25
    assigned = []
    for manager in managers:
        managed = store.get_customers(manager["manager_id"])
        assert len(managed) == 5
        assert {c["manager_id"] for c in managed} == {manager["manager_id"]}
        assigned.extend(c["customer_id"] for c in managed)
    assert sorted(assigned) == [c["customer_id"] for c in customers]
    assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store._connection.execute("PRAGMA foreign_key_check").fetchall() == []
    # No file-backed database is attached.
    assert store._connection.execute("PRAGMA database_list").fetchone()[2] == ""


def test_holdings_and_prices(store):
    prices = {p["symbol"]: p for p in store.get_market_prices()}
    asset_types = set()
    for customer in store.get_customers():
        holdings = store.get_holdings(customer["customer_id"])
        assert 5 <= len(holdings) <= 10
        assert len({h["symbol"] for h in holdings}) == len(holdings)
        for holding in holdings:
            assert holding["customer_id"] == customer["customer_id"]
            assert holding["symbol"] in prices
            assert holding["quantity"] > 0
            assert holding["average_cost"] > 0
            asset_types.add(holding["asset_type"])
    assert asset_types == {"stock", "broad_market_etf", "sector_etf", "bond_etf"}
    assert {p["as_of_date"] for p in prices.values()} == {PRICE_AS_OF_DATE}
    assert store.get_market_prices(["VTI", "AAPL", "AAPL", "UNKNOWN"]) == [
        {"symbol": "AAPL", "current_price": 336.64, "as_of_date": PRICE_AS_OF_DATE},
        {"symbol": "VTI", "current_price": 381.94, "as_of_date": PRICE_AS_OF_DATE},
    ]


def test_missing_identifiers_and_parameterized_queries(store):
    assert store.get_customer("missing") is None
    assert store.get_customers("missing") == []
    assert store.get_holdings("missing") == []
    assert store.get_market_prices([]) == []
    assert store.get_market_prices(["missing"]) == []
    injection = "' OR 1=1 --"
    assert store.get_customer(injection) is None
    assert store.get_customers(injection) == []
    assert store.get_holdings(injection) == []
    assert store.get_market_prices([injection]) == []


def snapshot(store):
    customers = store.get_customers()
    return (
        store.get_managers(),
        customers,
        store.get_market_prices(),
        [store.get_holdings(c["customer_id"]) for c in customers],
    )


def test_seed_is_identical_across_independent_instances(store):
    with PortfolioStore() as second:
        assert snapshot(store) == snapshot(second)
    with PortfolioStore() as recreated:
        assert snapshot(store) == snapshot(recreated)
    # Pin representative fixture values, not just equality between two executions.
    assert store.get_customer("C001") == {
        "customer_id": "C001",
        "customer_name": "Synthetic Customer 1: Diversified moderate",
        "manager_id": "PM001",
        "risk_profile": "moderate",
        "investment_horizon_years": 10,
        "investment_goal": "Long-term wealth",
        "liquidity_need": "low",
        "risk_profile_last_updated": "2025-12-01",
    }
    assert next(h for h in store.get_holdings("C002") if h["symbol"] == "AAPL") == {
        "customer_id": "C002",
        "symbol": "AAPL",
        "asset_type": "stock",
        "quantity": 237.64258555,
        "average_cost": 168.32,
        "sector": "Technology",
    }


def portfolio_metrics(store, customer_id):
    prices = {p["symbol"]: p["current_price"] for p in store.get_market_prices()}
    holdings = store.get_holdings(customer_id)
    values = [h["quantity"] * prices[h["symbol"]] for h in holdings]
    gains = [h["quantity"] * (prices[h["symbol"]] - h["average_cost"]) for h in holdings]
    total = sum(values)
    return {
        "total": total,
        "gain": sum(gains),
        "gains": gains,
        "largest_weight": max(values) / total,
        "bond_weight": sum(
            v for h, v in zip(holdings, values, strict=True) if h["asset_type"] == "bond_etf"
        )
        / total,
        "tech_weight": sum(
            v for h, v in zip(holdings, values, strict=True) if h["sector"] == "Technology"
        )
        / total,
    }


def test_deliberate_portfolios_have_meaningful_differences(store):
    for manager in store.get_managers():
        customers = store.get_customers(manager["manager_id"])
        moderate, concentrated, technology, conservative, aggressive = [
            portfolio_metrics(store, c["customer_id"]) for c in customers
        ]
        assert moderate["largest_weight"] < 0.30
        assert 0.30 < moderate["bond_weight"] < 0.50
        assert concentrated["largest_weight"] > 0.80
        assert concentrated["gain"] > concentrated["total"] * 0.40
        assert technology["tech_weight"] > 0.85
        assert min(technology["gains"]) < 0 < max(technology["gains"])
        assert conservative["bond_weight"] > 0.80
        assert conservative["gain"] < -conservative["total"] * 0.10
        assert aggressive["bond_weight"] == 0
        assert aggressive["largest_weight"] < 0.30
        assert aggressive["gain"] > aggressive["total"] * 0.25
        assert (
            len(
                {m["total"] for m in (moderate, concentrated, technology, conservative, aggressive)}
            )
            == 5
        )


def test_profile_data_is_read_only_and_connections_are_closed(store):
    before = store.get_customer("C001")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store._connection.execute(
            "UPDATE customers SET risk_profile = 'aggressive' WHERE customer_id = 'C001'"
        )
    assert store.get_customer("C001") == before
    with PortfolioStore() as temporary:
        assert temporary.get_managers()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        temporary.get_managers()


def test_frozen_october_9_closes_match_recorded_sources(store):
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "resources/portfolio_price_snapshot_2026-10-09.json"
    )
    reference = json.loads(path.read_text())
    assert reference["as_of_date"] == PRICE_AS_OF_DATE == "2026-10-09"
    assert reference["currency"] == "USD"
    expected = {p["symbol"]: p["close"] for p in reference["prices"]}
    assert len(expected) == 15
    assert {p["symbol"]: p["current_price"] for p in store.get_market_prices()} == expected
    assert all(p["source_url"].startswith("https://") for p in reference["prices"])
