"""Validate only HTTP calculation data; never read the portfolio store directly."""

import math

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finassist.web_app.services.models import QueryResponse


class Holding(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    symbol: str = Field(min_length=1)
    market_value: float = Field(ge=0)
    holding_allocation_pct: float = Field(ge=0, le=100)
    as_of_date: str | None = None
    sector: str | None = None
    asset_type: str | None = None
    unrealized_gain_loss: float | None = None


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    status: str
    customer_id: str
    customer_name: str | None = None
    data_source: str
    price_basis: str
    portfolio_value: float = Field(gt=0)
    holdings: list[Holding] = Field(min_length=1)
    manager_id: str | None = None
    concentration: dict[str, object] = Field(default_factory=dict)
    performance_complete: bool = False
    cost_basis: float | None = Field(default=None, gt=0)
    unrealized_gain_loss: float | None = None
    unrealized_gain_loss_pct: float | None = None

    @model_validator(mode="after")
    def consistent(self) -> "PortfolioSnapshot":
        if self.status != "success":
            raise ValueError("Not a successful portfolio calculation")
        if not self.data_source or not self.price_basis:
            raise ValueError("Missing provenance")
        if len({h.symbol for h in self.holdings}) != len(self.holdings):
            raise ValueError("Duplicate holdings")
        if not math.isclose(
            sum(h.market_value for h in self.holdings),
            self.portfolio_value,
            rel_tol=1e-6,
            abs_tol=0.02,
        ):
            raise ValueError("Holdings do not reconcile to valuation")
        for h in self.holdings:
            if abs(h.market_value / self.portfolio_value * 100 - h.holding_allocation_pct) > 0.02:
                raise ValueError("Allocation does not match holding values")
        if self.performance_complete and self.cost_basis is not None:
            expected_gain = self.portfolio_value - self.cost_basis
            if self.unrealized_gain_loss is not None and not math.isclose(
                self.unrealized_gain_loss, expected_gain, abs_tol=0.02, rel_tol=1e-6
            ):
                raise ValueError("Inconsistent performance data")
            if (
                self.unrealized_gain_loss_pct is not None
                and abs(self.unrealized_gain_loss_pct - expected_gain / self.cost_basis * 100)
                > 0.02
            ):
                raise ValueError("Inconsistent percentage gain")
        return self


def snapshots(response: QueryResponse | None) -> list[PortfolioSnapshot]:
    if (
        response is None
        or response.intent != "portfolio"
        or response.status in {"error", "clarification"}
    ):
        return []
    if response.clarification_question:
        return []
    items = ([response.calculation] if response.calculation is not None else []) + (
        response.calculations or []
    )
    results: dict[str, PortfolioSnapshot] = {}
    for item in items:
        candidates = item.get("portfolios", [item]) if item.get("status") == "success" else []
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            try:
                snapshot = PortfolioSnapshot.model_validate(candidate)
            except ValidationError:
                continue
            if (
                snapshot.data_source == "synthetic_test_data"
                and snapshot.price_basis == "frozen_historical_close_usd"
            ):
                results[snapshot.customer_id] = snapshot
    return list(results.values())


def sample_snapshot() -> PortfolioSnapshot:
    """UI-only fictional sample; not a copy of or lookup into the SQLite dataset."""
    return PortfolioSnapshot.model_validate(
        {
            "status": "success",
            "customer_id": "DEMO-ONLY",
            "data_source": "ui_synthetic_sample",
            "price_basis": "invented_illustrative_values",
            "portfolio_value": 100000,
            "holdings": [
                {
                    "symbol": "DEMO-STOCK",
                    "sector": "Example equity",
                    "asset_type": "Stocks",
                    "market_value": 60000,
                    "holding_allocation_pct": 60,
                },
                {
                    "symbol": "DEMO-BOND",
                    "sector": "Example fixed income",
                    "asset_type": "Bonds",
                    "market_value": 30000,
                    "holding_allocation_pct": 30,
                },
                {
                    "symbol": "DEMO-CASH",
                    "sector": "Example cash",
                    "asset_type": "Cash",
                    "market_value": 10000,
                    "holding_allocation_pct": 10,
                },
            ],
        }
    )


def fetch_current_portfolios() -> QueryResponse:
    """Use server-provided identity; no frontend identity or store access."""
    from finassist.web_app.config import WebSettings
    from finassist.web_app.services.api_client import FinanceApiClient
    from finassist.web_app.services.models import QueryRequest

    query = (
        "Load my portfolio landing dashboard using the trusted application identity. "
        "If I am an individual customer, retrieve my own complete portfolio snapshot. "
        "If I am a portfolio manager, retrieve and compare all customer portfolios assigned "
        "to me, with customer names and holdings. Do not screen or omit customers based on "
        "concentration. Include any excluded portfolios and reasons when data is unavailable. "
        "Do not ask me for my ID when application identity supplies it. "
        "Use portfolio tools and return their structured calculations; do not invent data."
    )
    with FinanceApiClient(WebSettings.from_environment()) as client:
        return client.query(QueryRequest(query=query))


def performance_label(snapshot: PortfolioSnapshot) -> str:
    """Describe holding-level unrealized results; mixed is not a net-return label."""
    gains = [holding.unrealized_gain_loss for holding in snapshot.holdings]
    if not snapshot.performance_complete or any(value is None for value in gains):
        return "Unavailable"
    positive = any(value is not None and value > 0 for value in gains)
    negative = any(value is not None and value < 0 for value in gains)
    if positive and negative:
        return "Mixed"
    if positive:
        return "Gains"
    if negative:
        return "Losses"
    return "Unchanged"
