"""Deterministic extrema over validated provider observations, never model-supplied prices."""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from mcp import ClientSession

from finassist.agents.market_analysis.cache import market_cache
from finassist.agents.market_analysis.chart_data import chart_observation
from finassist.agents.market_analysis.growth import RATE_LIMIT_RESPONSE, is_rate_limit
from finassist.workflow.state import FinanceAssistantState


def compute_price_extrema(
    rows: list[dict[str, Any]],
    start: str,
    end: str,
    operation: Literal["min", "max"],
    field: str,
) -> dict[str, Any]:
    if operation not in {"min", "max"} or field not in {"open", "high", "low", "close"}:
        raise ValueError("Unsupported operation or price field.")
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    if start_date > end_date:
        raise ValueError("Invalid date range.")
    values: dict[str, Decimal] = {}
    for row in rows:
        try:
            day = str(row["datetime"])[:10]
            observed = date.fromisoformat(day)
            if not start_date <= observed <= end_date:
                continue
            price = Decimal(str(row[field]))
            if not price.is_finite() or price <= 0:
                raise ValueError("Invalid provider price.")
            if all(row.get(key) is not None for key in ("open", "high", "low", "close")):
                opening, high, low, close = (
                    Decimal(str(row[key])) for key in ("open", "high", "low", "close")
                )
                if any(
                    not value.is_finite() or value <= 0 for value in (opening, high, low, close)
                ) or not (low <= min(opening, close) <= max(opening, close) <= high):
                    raise ValueError("Invalid OHLC range.")
            if day in values and values[day] != price:
                raise ValueError("Conflicting provider observations.")
            values[day] = price
        except (KeyError, InvalidOperation, TypeError) as error:
            raise ValueError("Missing or invalid provider observation.") from error
    if not values:
        raise ValueError("No observations in the requested range.")
    extreme = (min if operation == "min" else max)(values.values())
    return {
        "type": "market_price_extrema",
        "operation": operation,
        "field": field,
        "value": str(extreme),
        "observation_dates": sorted(day for day, price in values.items() if price == extreme),
        "requested_start_date": start,
        "requested_end_date": end,
        "observed_start_date": min(values),
        "observed_end_date": max(values),
        "observation_count": len(values),
        "coverage": "unverified",
    }


async def calculate_price_extrema(
    session: ClientSession,
    symbol: str,
    parameters: dict[str, Any],
    operation: Literal["min", "max"],
    field: str | None = None,
) -> FinanceAssistantState:
    field = field or ("high" if operation == "max" else "low")
    try:
        symbol = symbol.strip().upper()
        start, end = parameters["start_date"], parameters["end_date"]
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        if not symbol or not 1 <= days <= 100:
            raise ValueError("Use a date range of at most 100 calendar days.")
        result = await market_cache.call_tool(
            session,
            "get_time_series",
            {
                "symbol": symbol,
                "interval": "1day",
                "start_date": start,
                "end_date": end,
                "outputsize": 100,
            },
            retry=False,
        )
        if is_rate_limit(result):
            return {"answer": RATE_LIMIT_RESPONSE, "status": "error"}
        observation = chart_observation(
            result, tool="get_time_series", symbol=symbol, interval="1day"
        )
        if not observation or observation["symbol"].upper() != symbol:
            raise ValueError("No matching provider history.")
        calculation = compute_price_extrema(observation["values"], start, end, operation, field)
        calculation.update(
            {"symbol": symbol, "currency": observation.get("currency"), "provider": "Twelve Data"}
        )
        note = (
            "Computed from returned daily observations only; full trading-session coverage "
            "is not verified, and the current day's bar may be incomplete."
        )
        currency = observation.get("currency") or "currency not supplied"
        answer = (
            f"{symbol}: the {'highest' if operation == 'max' else 'lowest'} observed "
            f"{field} was {calculation['value']} ({currency}) on "
            f"{', '.join(calculation['observation_dates'])}. {note}"
        )
        return {"answer": answer, "status": "answered", "calculation": calculation}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {
            "answer": "Could not establish a historical price extreme from valid provider data. "
            "Try an explicit date range of at most 100 calendar days.",
            "status": "insufficient",
        }
