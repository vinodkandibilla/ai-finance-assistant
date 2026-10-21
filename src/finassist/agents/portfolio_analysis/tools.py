"""Model-callable retrieval and deterministic metrics over synthetic SQLite data."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from finassist.agents.portfolio_analysis.authorization import UserContext, denied
from finassist.agents.portfolio_analysis.authorization import (
    customer_id as normalize_customer_id,
)
from finassist.data.portfolio_store import PRICE_BASIS, PortfolioStore

RankBy = Literal["concentration", "diversification", "unrealized_loss"]
Screen = Literal["all", "concentrated", "unrealized_losses"]


@dataclass(frozen=True)
class ConcentrationThresholds:
    """Percentage thresholds; flags use strict > comparisons before rounding."""

    holding_pct: float = 30.0
    sector_pct: float = 50.0

    def __post_init__(self) -> None:
        for value in (self.holding_pct, self.sector_pct):
            if not 0 <= value <= 100:
                raise ValueError(
                    "Concentration thresholds must be finite percentages from 0 to 100"
                )


def _number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and result >= 0 else None
    except (InvalidOperation, ValueError):
        return None


def _display(value: Decimal | None) -> float | None:
    return float(round(value, 2)) if value is not None else None


def calculate_portfolio(
    holdings: list[dict[str, Any]],
    prices: list[dict[str, Any]],
    thresholds: ConcentrationThresholds | None = None,
) -> dict[str, Any]:
    """Calculate long-only snapshot metrics; no ETF constituent look-through.

    Missing/invalid prices or quantities prevent partial valuation. Invalid/zero
    costs preserve allocation but leave affected gain metrics unavailable.
    """
    limits = thresholds or ConcentrationThresholds()
    if not holdings:
        return {"status": "insufficient", "code": "EMPTY_PORTFOLIO"}
    quotes = {p["symbol"]: p for p in prices}
    rows = []
    warnings = []
    for holding in sorted(holdings, key=lambda h: h["symbol"]):
        symbol = holding["symbol"]
        quote = quotes.get(symbol, {})
        quantity = _number(holding.get("quantity"))
        price = _number(quote.get("current_price"))
        if quantity is None or price is None:
            return {"status": "insufficient", "code": "INVALID_VALUATION_DATA", "symbol": symbol}
        value = quantity * price
        cost = _number(holding.get("average_cost"))
        basis = quantity * cost if cost is not None and cost > 0 and quantity > 0 else None
        gain = value - basis if basis is not None else None
        if basis is None:
            warnings.append({"symbol": symbol, "code": "INVALID_COST_BASIS"})
        rows.append(
            {
                **holding,
                "current_price": price,
                "as_of_date": quote.get("as_of_date"),
                "market_value": value,
                "cost_basis": basis,
                "unrealized_gain_loss": gain,
                "unrealized_gain_loss_pct": gain / basis * 100
                if basis is not None and gain is not None
                else None,
            }
        )
    total = sum((r["market_value"] for r in rows), Decimal(0))
    if total <= 0:
        return {"status": "insufficient", "code": "ZERO_PORTFOLIO_VALUE"}
    sectors: dict[str, Decimal] = {}
    classes: dict[str, Decimal] = {}
    for row in rows:
        row["holding_allocation_pct"] = row["market_value"] / total * 100
        sectors[row["sector"]] = sectors.get(row["sector"], Decimal(0)) + row["market_value"]
        classes[row["asset_type"]] = (
            classes.get(row["asset_type"], Decimal(0)) + row["market_value"]
        )
    largest = min(rows, key=lambda r: (-r["market_value"], r["symbol"]))
    largest_sector = min(sectors, key=lambda s: (-sectors[s], s))
    sector_pct = sectors[largest_sector] / total * 100
    holding_limit = Decimal(str(limits.holding_pct))
    sector_limit = Decimal(str(limits.sector_pct))
    holding_flags = [r["symbol"] for r in rows if r["holding_allocation_pct"] > holding_limit]
    sector_flags = sorted(s for s, v in sectors.items() if v / total * 100 > sector_limit)
    complete_costs = not warnings
    basis = sum((r["cost_basis"] for r in rows), Decimal(0)) if complete_costs else None
    gain = total - basis if basis is not None else None
    # HHI is a holdings-weight concentration measure, not a risk score.
    hhi = sum(((r["market_value"] / total) ** 2 for r in rows), Decimal(0))

    def render(row: dict[str, Any]) -> dict[str, Any]:
        return {k: _display(v) if isinstance(v, Decimal) else v for k, v in row.items()}

    def allocation(values: dict[str, Decimal]) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "market_value": _display(value),
                "allocation_pct": _display(value / total * 100),
            }
            for name, value in sorted(values.items())
        ]

    winners = sorted(
        (
            r
            for r in rows
            if r["unrealized_gain_loss"] is not None and r["unrealized_gain_loss"] > 0
        ),
        key=lambda r: (-r["unrealized_gain_loss"], r["symbol"]),
    )
    losers = sorted(
        (
            r
            for r in rows
            if r["unrealized_gain_loss"] is not None and r["unrealized_gain_loss"] < 0
        ),
        key=lambda r: (r["unrealized_gain_loss"], r["symbol"]),
    )
    return {
        "status": "success",
        "portfolio_value": _display(total),
        "cost_basis": _display(basis),
        "unrealized_gain_loss": _display(gain),
        "unrealized_gain_loss_pct": _display(gain / basis * 100)
        if basis is not None and gain is not None
        else None,
        "performance_complete": complete_costs,
        "warnings": warnings,
        "holdings": [render(r) for r in rows],
        "sector_allocation": allocation(sectors),
        "asset_class_allocation": allocation(classes),
        "largest_holding": render(largest),
        "largest_sector": {"name": largest_sector, "allocation_pct": _display(sector_pct)},
        "concentration": {
            "holding_threshold_pct": limits.holding_pct,
            "sector_threshold_pct": limits.sector_pct,
            "holdings_over_threshold": holding_flags,
            "sectors_over_threshold": sector_flags,
            "is_concentrated": bool(holding_flags or sector_flags),
        },
        "diversification": {
            "holding_count": len(rows),
            "sector_count": len(sectors),
            "asset_class_count": len(classes),
            "holding_hhi": float(hhi),
            "effective_holding_count": float(1 / hhi),
        },
        "biggest_gainers": [render(r) for r in winners],
        "biggest_losers": [render(r) for r in losers],
        "assumptions": [
            "Synthetic portfolios with frozen reference closes; "
            "gains are unrealized, not historical returns.",
            "Sector and asset-class allocations use instrument labels; no ETF look-through.",
            "Gainers/losers rank by monetary gain/loss, not percentage.",
            "HHI uses holding weights; lower HHI means greater holding diversification.",
        ],
    }


class PortfolioTools:
    """Inject a store, thresholds and application identity; deny unauthorized access."""

    def __init__(
        self,
        store: PortfolioStore,
        thresholds: ConcentrationThresholds | None = None,
        user_context: UserContext | None = None,
    ):
        self.user_context = user_context
        self.store = store
        self.thresholds = thresholds or ConcentrationThresholds()

    def _allowed_customer(self, identifier: str) -> bool:
        user = self.user_context
        if user is None:
            return False
        if user.role == "CUSTOMER":
            return identifier == normalize_customer_id(user.user_id)
        customer = self.store.get_customer(identifier)
        return customer is not None and customer["manager_id"] == user.user_id

    def _allowed_manager(self, manager_id: str) -> bool:
        user = self.user_context
        return user is not None and user.role == "PORTFOLIO_MANAGER" and user.user_id == manager_id

    def resolve_customer(self, name: str) -> dict[str, Any]:
        user = self.user_context
        if user is None:
            return denied()
        candidates = (
            self.store.get_customers(user.user_id)
            if user.role == "PORTFOLIO_MANAGER"
            else [self.store.get_customer(normalize_customer_id(user.user_id))]
        )
        matches = [
            {"customer_id": c["customer_id"], "customer_name": c["customer_name"]}
            for c in candidates
            if c and name.strip() and name.casefold() in c["customer_name"].casefold()
        ]
        return {"status": "success", "matches": matches, "ambiguous": len(matches) > 1}

    def get_customer_portfolio(self, customer_id: str) -> dict[str, Any]:
        customer_id = normalize_customer_id(customer_id)
        if not self._allowed_customer(customer_id):
            return denied()
        customer = self.store.get_customer(customer_id)
        if customer is None:
            return {"status": "error", "code": "CUSTOMER_NOT_FOUND", "customer_id": customer_id}
        holdings = self.store.get_holdings(customer_id)
        result = calculate_portfolio(
            holdings, self.store.get_market_prices([h["symbol"] for h in holdings]), self.thresholds
        )
        return {
            **result,
            "customer_id": customer_id,
            "customer_name": customer["customer_name"],
            "manager_id": customer["manager_id"],
            "data_source": self.store.data_source,
            "price_basis": PRICE_BASIS,
        }

    def get_customer_investment_profile(self, customer_id: str) -> dict[str, Any]:
        customer_id = normalize_customer_id(customer_id)
        if not self._allowed_customer(customer_id):
            return denied()
        customer = self.store.get_customer(customer_id)
        if customer is None:
            return {"status": "error", "code": "CUSTOMER_NOT_FOUND", "customer_id": customer_id}
        return {
            "status": "success",
            "data_source": self.store.data_source,
            "price_basis": PRICE_BASIS,
            "profile": customer,
            "risk_profile_source": "pre_established",
        }

    def compare_manager_portfolios(
        self,
        manager_id: str,
        customer_ids: list[str] | None = None,
        rank_by: RankBy = "concentration",
    ) -> dict[str, Any]:
        if not self._allowed_manager(manager_id):
            return denied()
        if customer_ids is not None:
            customer_ids = [normalize_customer_id(cid) for cid in customer_ids]
            if not all(self._allowed_customer(cid) for cid in customer_ids):
                return denied()
        if rank_by not in {"concentration", "diversification", "unrealized_loss"}:
            return {"status": "error", "code": "INVALID_RANK_BY"}
        if not any(m["manager_id"] == manager_id for m in self.store.get_managers()):
            return {"status": "error", "code": "MANAGER_NOT_FOUND"}
        customers = self.store.get_customers(manager_id)
        available = {c["customer_id"] for c in customers}
        if customer_ids is not None and not set(customer_ids) <= available:
            return {"status": "error", "code": "CUSTOMER_NOT_IN_MANAGER_DATASET"}
        selected = available if customer_ids is None else set(customer_ids)
        results = [self.get_customer_portfolio(cid) for cid in sorted(selected)]
        valid = [r for r in results if r["status"] == "success"]
        excluded = [r for r in results if r["status"] != "success"]
        if rank_by == "unrealized_loss":
            excluded.extend(r for r in valid if not r["performance_complete"])
            valid = [r for r in valid if r["performance_complete"]]
        if rank_by == "concentration":
            valid.sort(key=lambda r: (-r["diversification"]["holding_hhi"], r["customer_id"]))
        elif rank_by == "diversification":
            valid.sort(key=lambda r: (r["diversification"]["holding_hhi"], r["customer_id"]))
        else:
            valid.sort(key=lambda r: (r["unrealized_gain_loss"], r["customer_id"]))
        return {
            "status": "success",
            "manager_id": manager_id,
            "rank_by": rank_by,
            "ranking_metric": "unrealized_gain_loss_ascending"
            if rank_by == "unrealized_loss"
            else "holding_hhi",
            "portfolios": valid,
            "excluded_portfolios": excluded,
            "data_source": self.store.data_source,
            "price_basis": PRICE_BASIS,
        }

    def screen_manager_portfolios(
        self,
        manager_id: str,
        screen: Screen = "concentrated",
        rank_by: RankBy = "concentration",
    ) -> dict[str, Any]:
        if screen not in {"all", "concentrated", "unrealized_losses"}:
            return {"status": "error", "code": "INVALID_SCREEN"}
        result = self.compare_manager_portfolios(manager_id, rank_by=rank_by)
        if result["status"] != "success":
            return result
        if screen == "concentrated":
            result["portfolios"] = [
                r for r in result["portfolios"] if r["concentration"]["is_concentrated"]
            ]
        elif screen == "unrealized_losses":
            result["portfolios"] = [
                r
                for r in result["portfolios"]
                if r["unrealized_gain_loss"] is not None and r["unrealized_gain_loss"] < 0
            ]
        return {**result, "screen": screen}

    def as_tools(self) -> list[BaseTool]:
        def scoped(config: RunnableConfig) -> "PortfolioTools":
            raw = config.get("configurable", {}).get("portfolio_user_context")
            user = UserContext.model_validate(raw) if raw is not None else self.user_context
            return PortfolioTools(self.store, self.thresholds, user)

        @tool
        async def get_customer_portfolio(
            config: RunnableConfig, customer_id: str | None = None
        ) -> dict[str, Any]:
            """Retrieve authorized metrics; omitted ID means the current customer's own."""
            tools = scoped(config)
            user = tools.user_context
            identifier = customer_id or (user.user_id if user and user.role == "CUSTOMER" else "")
            if not identifier:
                return (
                    {"status": "insufficient", "code": "CUSTOMER_ID_REQUIRED"} if user else denied()
                )
            return tools.get_customer_portfolio(identifier)

        @tool
        async def get_customer_investment_profile(
            config: RunnableConfig, customer_id: str | None = None
        ) -> dict[str, Any]:
            """Read authorized stored profile; omitted ID means the current customer's own."""
            tools = scoped(config)
            user = tools.user_context
            identifier = customer_id or (user.user_id if user and user.role == "CUSTOMER" else "")
            if not identifier:
                return (
                    {"status": "insufficient", "code": "CUSTOMER_ID_REQUIRED"} if user else denied()
                )
            return tools.get_customer_investment_profile(identifier)

        @tool
        async def screen_manager_portfolios(
            config: RunnableConfig,
            manager_id: str | None = None,
            screen: Screen = "concentrated",
            rank_by: RankBy = "concentration",
            finish_response: bool = False,
        ) -> dict[str, Any]:
            """Screen authorized customers; omitted manager means the current manager.

            Set finish_response true only when this screen fully answers a routine
            listing request; false for interpretation, profile alignment or additional tasks.
            Python formats successful completed screens without another model call.
            """
            tools = scoped(config)
            user = tools.user_context
            return tools.screen_manager_portfolios(
                manager_id or (user.user_id if user else ""), screen, rank_by
            )

        @tool
        async def compare_manager_portfolios(
            config: RunnableConfig,
            manager_id: str | None = None,
            customer_ids: list[str] | None = None,
            rank_by: RankBy = "concentration",
            finish_response: bool = False,
        ) -> dict[str, Any]:
            """Compare authorized customers; omitted manager means the current manager.

            Set finish_response true for a complete routine listing or dashboard;
            false for interpretation, profile alignment or additional tasks.
            Python formats successful completed comparisons without another model call.
            """
            tools = scoped(config)
            user = tools.user_context
            return tools.compare_manager_portfolios(
                manager_id or (user.user_id if user else ""), customer_ids, rank_by
            )

        @tool
        async def resolve_portfolio_customer(name: str, config: RunnableConfig) -> dict[str, Any]:
            """Find customers by name only in authorized scope; clarify if multiple matches."""
            return scoped(config).resolve_customer(name)

        return [
            get_customer_portfolio,
            get_customer_investment_profile,
            screen_manager_portfolios,
            compare_manager_portfolios,
            resolve_portfolio_customer,
        ]
