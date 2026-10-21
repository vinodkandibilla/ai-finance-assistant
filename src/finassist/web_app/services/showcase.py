"""Deterministic goal scenarios and validated display data, independent of Streamlit."""

import math
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finassist.web_app.services.models import QueryResponse


@dataclass(frozen=True)
class Projection:
    balances: tuple[float, ...]
    contributed: tuple[float, ...]
    required_monthly: float
    shortfall: float


def project_goal(
    target: float, savings: float, contribution: float, annual_return: float, months: int
) -> Projection:
    if (
        any(not math.isfinite(x) or x < 0 for x in [target, savings, contribution, annual_return])
        or not 1 <= months <= 1200
        or isinstance(months, bool)
        or not isinstance(months, int)
        or annual_return > 100
    ):
        raise ValueError("Use finite nonnegative inputs, 1–1200 months, and a return at most 100%.")
    rate = annual_return / 1200
    balances = [savings]
    contributed = [savings]
    for month in range(1, months + 1):
        balances.append(balances[-1] * (1 + rate) + contribution)
        contributed.append(savings + month * contribution)
    factor = math.expm1(months * math.log1p(rate)) / rate if rate else months
    growth = math.exp(months * math.log1p(rate)) if rate else 1
    required = max(0, (target - savings * growth) / factor)
    if any(not math.isfinite(v) for v in [*balances, *contributed, required]):
        raise ValueError("Projection exceeds supported range.")
    return Projection(
        tuple(round(v, 2) for v in balances),
        tuple(round(v, 2) for v in contributed),
        math.ceil(round(required * 100, 8)) / 100,
        round(max(0, target - balances[-1]), 2),
    )


class PriceEndpoints(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    symbol: str = Field(min_length=1)
    currency: str = "Not supplied"
    start_date: str
    end_date: str
    start_price: float = Field(gt=0)
    end_price: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_dates(self) -> "PriceEndpoints":
        from datetime import date

        if date.fromisoformat(self.start_date) > date.fromisoformat(self.end_date):
            raise ValueError("Dates out of order")
        return self


def market_endpoints(response: QueryResponse | None) -> list[PriceEndpoints]:
    if not response or response.intent != "market" or response.status in {"error", "clarification"}:
        return []
    if response.clarification_question:
        return []
    items = ([response.calculation] if response.calculation is not None else []) + (
        response.calculations or []
    )
    result = []
    for item in items:
        try:
            result.append(PriceEndpoints.model_validate(item))
        except ValidationError:
            continue
    return result


def safe_source(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or any(ord(c) < 32 for c in url)
        ):
            return None
        return parsed.hostname
    except ValueError:
        return None


def scope_market_question(question: str) -> str:
    """Constrain Market-page requests without adding unsupported API fields."""
    return (
        "Market scope: United States, NASDAQ-listed equities only, currency USD. "
        "No depositary receipts. Keep answer under 120 words; at most three symbol matches. "
        "Omit OHLC, volume and 52-week statistics unless requested.\nQuestion: " + question
    )
