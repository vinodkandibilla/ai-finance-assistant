import pytest
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from ai_finance_assistant.agents.portfolio_analysis.authorization import UserContext
from ai_finance_assistant.agents.portfolio_analysis.store import PortfolioStore
from ai_finance_assistant.agents.portfolio_analysis.tools import (
    ConcentrationThresholds,
    PortfolioTools,
    calculate_portfolio,
)


def fixture_portfolio(cost=8):
    holdings = [
        {
            "symbol": "A",
            "quantity": 3,
            "average_cost": cost,
            "sector": "Tech",
            "asset_type": "stock",
        },
        {"symbol": "B", "quantity": 2, "average_cost": 12, "sector": "Tech", "asset_type": "stock"},
        {
            "symbol": "C",
            "quantity": 5,
            "average_cost": 10,
            "sector": "Bonds",
            "asset_type": "bond_etf",
        },
    ]
    prices = [{"symbol": s, "current_price": 10, "as_of_date": "2026-10-09"} for s in "ABC"]
    return holdings, prices


def test_exact_metrics_and_strict_thresholds():
    result = calculate_portfolio(*fixture_portfolio())
    assert result["portfolio_value"] == 100
    assert result["cost_basis"] == 98
    assert result["unrealized_gain_loss"] == 2
    assert result["unrealized_gain_loss_pct"] == 2.04
    a = result["holdings"][0]
    assert (
        a["market_value"],
        a["cost_basis"],
        a["unrealized_gain_loss"],
        a["unrealized_gain_loss_pct"],
        a["holding_allocation_pct"],
    ) == (30, 24, 6, 25, 30)
    assert result["concentration"]["holdings_over_threshold"] == ["C"]
    assert result["concentration"]["sectors_over_threshold"] == []
    assert result["largest_holding"]["symbol"] == "C"
    assert result["largest_sector"]["name"] == "Bonds"  # Stable alphabetical tie.
    assert result["biggest_gainers"][0]["symbol"] == "A"
    assert result["biggest_losers"][0]["symbol"] == "B"
    assert result["diversification"]["holding_hhi"] == pytest.approx(0.38)
    assert result["diversification"]["effective_holding_count"] == pytest.approx(1 / 0.38)
    assert sum(r["allocation_pct"] for r in result["sector_allocation"]) == 100
    assert sum(r["allocation_pct"] for r in result["asset_class_allocation"]) == 100
    custom = calculate_portfolio(*fixture_portfolio(), ConcentrationThresholds(50, 49))
    assert custom["concentration"]["holdings_over_threshold"] == []
    assert custom["concentration"]["sectors_over_threshold"] == ["Bonds", "Tech"]


@pytest.mark.parametrize("cost", [0, -1, None, float("nan"), float("inf"), "invalid"])
def test_invalid_cost_does_not_fabricate_performance(cost):
    result = calculate_portfolio(*fixture_portfolio(cost))
    assert result["portfolio_value"] == 100
    assert result["cost_basis"] is None
    assert result["unrealized_gain_loss"] is None
    assert result["unrealized_gain_loss_pct"] is None
    assert result["performance_complete"] is False
    assert result["holdings"][0]["unrealized_gain_loss"] is None
    assert result["warnings"] == [{"symbol": "A", "code": "INVALID_COST_BASIS"}]


@pytest.mark.parametrize("value", [-1, None, float("nan"), float("inf")])
def test_invalid_prices_prevent_partial_valuation(value):
    holdings, prices = fixture_portfolio()
    prices[0]["current_price"] = value
    assert calculate_portfolio(holdings, prices)["code"] == "INVALID_VALUATION_DATA"


def test_empty_missing_and_zero_values():
    holdings, prices = fixture_portfolio()
    assert calculate_portfolio([], prices)["code"] == "EMPTY_PORTFOLIO"
    assert calculate_portfolio(holdings, prices[1:])["code"] == "INVALID_VALUATION_DATA"
    for h in holdings:
        h["quantity"] = 0
    assert calculate_portfolio(holdings, prices)["code"] == "ZERO_PORTFOLIO_VALUE"


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf")])
def test_threshold_validation(value):
    with pytest.raises(ValueError):
        ConcentrationThresholds(holding_pct=value)


def test_manager_screening_rankings_and_profile():
    with PortfolioStore() as store:
        tools = PortfolioTools(
            store, user_context=UserContext(user_id="PM001", role="PORTFOLIO_MANAGER")
        )
        profile = tools.get_customer_investment_profile("C001")
        assert profile["profile"]["risk_profile"] == "moderate"
        assert profile["risk_profile_source"] == "pre_established"
        ranked = tools.compare_manager_portfolios("PM001")
        assert ranked["portfolios"][0]["customer_id"] == "C002"
        diversified = tools.compare_manager_portfolios("PM001", rank_by="diversification")
        assert diversified["portfolios"][0]["customer_id"] == "C005"
        losses = tools.screen_manager_portfolios(
            "PM001", screen="unrealized_losses", rank_by="unrealized_loss"
        )
        assert [r["customer_id"] for r in losses["portfolios"]] == ["C004"]
        concentrated = tools.screen_manager_portfolios("PM001")
        assert {r["customer_id"] for r in concentrated["portfolios"]} == {"C002", "C003", "C004"}
        subset = tools.compare_manager_portfolios("PM001", ["C003", "C002", "C002"])
        assert [r["customer_id"] for r in subset["portfolios"]] == ["C002", "C003"]
        assert tools.compare_manager_portfolios("PM001", ["C006"])["status"] == "error"
        assert tools.compare_manager_portfolios("missing")["code"] == "ACCESS_DENIED"
        assert tools.get_customer_portfolio("missing")["code"] == "ACCESS_DENIED"
        assert tools.get_customer_investment_profile("missing")["code"] == "ACCESS_DENIED"
        assert tools.compare_manager_portfolios("PM001", [])["portfolios"] == []
        assert tools.compare_manager_portfolios("PM001") == ranked
        assert tools.get_customer_investment_profile("C001") == profile


async def test_langgraph_async_tool_execution_uses_injected_store():
    with PortfolioStore() as store:
        node = ToolNode(
            PortfolioTools(
                store, user_context=UserContext(user_id="PM001", role="PORTFOLIO_MANAGER")
            ).as_tools()
        )
        result = await node.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_customer_portfolio",
                                "args": {"customer_id": "C001"},
                                "id": "one",
                            },
                            {
                                "name": "get_customer_investment_profile",
                                "args": {"customer_id": "C001"},
                                "id": "two",
                            },
                        ],
                    )
                ]
            }
        )
        import json

        payloads = [json.loads(m.content) for m in result["messages"]]
        assert all(p["status"] == "success" for p in payloads)
        assert payloads[0]["portfolio_value"] == 69500
