"""HTTP-only market dashboard data and strict chart validation."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finassist.web_app.config import WebSettings
from finassist.web_app.services.api_client import FinanceApiClient
from finassist.web_app.services.models import QueryRequest, QueryResponse


class MarketBar(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    datetime: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_bar(self) -> "MarketBar":
        datetime.fromisoformat(self.datetime.replace("Z", "+00:00"))
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("Invalid OHLC range")
        return self


def comparison_series(
    first: list[MarketBar], second: list[MarketBar]
) -> tuple[list[str], list[float], list[float]]:
    """Compare closing prices on shared dates, rebased to 100 at the first date."""
    first_prices = {bar.datetime[:10]: bar.close for bar in first}
    second_prices = {bar.datetime[:10]: bar.close for bar in second}
    dates = sorted(first_prices.keys() & second_prices.keys())
    if len(dates) < 2:
        return [], [], []
    return (
        dates,
        [100 * first_prices[day] / first_prices[dates[0]] for day in dates],
        [100 * second_prices[day] / second_prices[dates[0]] for day in dates],
    )


def fetch_market_quote(symbol: str) -> QueryResponse:
    """Request only a quote via the public query endpoint."""
    with FinanceApiClient(WebSettings.from_environment()) as client:
        return client.query(
            QueryRequest(
                query=(
                    f"Retrieve the latest available quote for {symbol}. "
                    "Scope: United States, NASDAQ-listed equities only, currency USD. "
                    "Exclude all other exchanges. Include open, high, low, close, volume, "
                    "previous close and provider observation time. Do not fetch time series. "
                    "Keep the summary under 120 words. Do not invent missing data."
                )
            )
        )


def fetch_market_snapshot(symbol: str, interval: str) -> QueryResponse:
    window = "1 month" if interval == "Daily" else "3 months"
    query = (
        "Scope: United States NASDAQ-listed equities only, currency USD. "
        "Exclude international listings and depositary receipts. "
        f"Retrieve the latest available quote for {symbol}, NASDAQ, USD, including open, high, "
        f"low, close, volume, previous close and provider observation time. Also retrieve "
        f"{interval.lower()} OHLC time series for {symbol} over the last {window}. "
        "Use available observations only; do not invent unavailable prices."
        " Keep the textual summary under 120 words. Do not reproduce historical rows "
        "in prose or tables; the chart uses the structured provider observations."
    )
    with FinanceApiClient(WebSettings.from_environment()) as client:
        return client.query(QueryRequest(query=query))


def market_chart_data(
    response: QueryResponse, symbol: str, interval: str
) -> tuple[list[MarketBar], dict[str, Any]]:
    if response.status in {"error", "clarify", "clarification", "insufficient"}:
        return [], {}
    bars: list[MarketBar] = []
    quote: dict[str, Any] = {}
    for item in ([response.calculation] if response.calculation else []) + (
        response.calculations or []
    ):
        if item.get("type") != "market_observations" or item.get("symbol") != symbol:
            continue
        if item.get("currency") != "USD" or item.get("exchange") not in {"NASDAQ", "XNAS"}:
            continue
        rows = item.get("values")
        if not isinstance(rows, list):
            continue
        if item.get("interval") == "quote" and rows and isinstance(rows[0], dict):
            quote = rows[0]
        if item.get("interval") != {"Daily": "1day", "Weekly": "1week"}[interval]:
            continue
        for row in rows:
            try:
                bar = MarketBar.model_validate(row)
                if date.fromisoformat(bar.datetime[:10]) <= date.today():
                    bars.append(bar)
            except (ValidationError, ValueError):
                continue
    return sorted({bar.datetime: bar for bar in bars}.values(), key=lambda bar: bar.datetime), quote
