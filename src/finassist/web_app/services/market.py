"""HTTP-only market dashboard data and strict chart validation."""

import re
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
                market_scope="us_nasdaq_usd",
                query=(
                    f"Latest quote only for {symbol}: OHLC, volume, previous close and "
                    "provider observation time. Keep summary under 120 words."
                )
            )
        )


def fetch_market_snapshot(symbol: str | list[str], interval: str) -> QueryResponse:
    window = "1 month" if interval == "Daily" else "3 months"
    target = symbol if isinstance(symbol, str) else " and ".join(symbol)
    task = (
        f"Show {target}'s latest quote"
        if isinstance(symbol, str)
        else f"For each of {target}, show latest quote"
    )
    query = f"{task} and {interval.lower()} OHLC chart for the last {window}."
    with FinanceApiClient(WebSettings.from_environment()) as client:
        return client.query(QueryRequest(query=query, market_scope="us_nasdaq_usd"))


def resolved_market_symbol(response: QueryResponse) -> str | None:
    """Accept a single NASDAQ/USD symbol from successful structured observations."""
    if response.intent != "market" or response.status not in {None, "answered"}:
        return None
    symbols = {
        item["symbol"]
        for item in ([response.calculation] if response.calculation else [])
        + (response.calculations or [])
        if item.get("type") == "market_observations"
        and item.get("currency") == "USD"
        and item.get("exchange") in {"NASDAQ", "XNAS"}
        and isinstance(item.get("symbol"), str)
        and re.fullmatch(r"[A-Z][A-Z0-9.:-]{0,19}", item["symbol"])
    }
    return next(iter(symbols)) if len(symbols) == 1 else None


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
