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


def fetch_market_snapshot(symbol: str, interval: str) -> QueryResponse:
    query = (
        "Scope: United States NASDAQ-listed equities only, currency USD. "
        "Exclude international listings and depositary receipts. "
        f"Retrieve the latest available quote for {symbol}, NASDAQ, USD, including open, high, "
        f"low, close, volume, previous close and provider observation time. Also retrieve "
        f"{interval.lower()} OHLC time series for {symbol} over the last 3 months. "
        "Use available observations only; do not invent unavailable prices."
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
