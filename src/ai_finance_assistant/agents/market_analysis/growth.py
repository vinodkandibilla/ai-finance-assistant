"""Deterministic price growth and comparisons using bounded MCP requests."""

import asyncio
import csv
import io
import json
import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from mcp import ClientSession

from ai_finance_assistant.agents.market_analysis.cache import market_cache
from ai_finance_assistant.workflow.state import FinanceAssistantState

MAX_COMPARISON_ASSETS = 4
RATE_LIMIT_RESPONSE = (
    "The Market Analysis agent's API rate limit was reached. Please try again after one minute."
)


def is_rate_limit(value: object) -> bool:
    text = str(value).lower()
    return bool(
        re.search(
            r"\b429\b|rate[ _-]?limit|too many requests|"
            r"api credits.*(?:exceed|limit|exhaust)|"
            r"(?:exceed|exhaust).*api credits",
            text,
        )
    )


def compute_price_growth(start: dict[str, str], end: dict[str, str]) -> dict[str, str]:
    """Calculate price growth from provider-selected observations."""
    baseline, current = Decimal(start["close"]), Decimal(end["close"])
    if not baseline.is_finite() or not current.is_finite() or baseline <= 0 or current < 0:
        raise ValueError("Growth requires a positive baseline and a finite nonnegative end price.")
    change = current - baseline
    growth = change / baseline * 100
    return {
        "start_date": start["datetime"],
        "end_date": end["datetime"],
        "start_price": str(baseline),
        "end_price": str(current),
        "price_change": str(change),
        "growth_percent": str(growth),
        "math": f"({current} - {baseline}) / {baseline} × 100 = {growth:.2f}%",
    }


async def calculate_price_growth(
    session: ClientSession, symbol: str | list[str], parameters: dict[str, Any], as_of: str
) -> FinanceAssistantState:
    """Reuse the growth tool for one asset or a comparison over common observation dates."""
    symbols = list(
        dict.fromkeys(
            asset.strip().upper() for asset in ([symbol] if isinstance(symbol, str) else symbol)
        )
    )
    if not symbols or any(not asset for asset in symbols):
        return {"answer": "Please specify the ticker symbols to evaluate.", "status": "error"}
    if len(symbols) > MAX_COMPARISON_ASSETS:
        question = "Please compare no more than four assets at a time. Which four should I use?"
        return {"answer": question, "clarification_question": question, "status": "clarify"}
    comparison = len(symbols) > 1

    async def records(name: str, arguments: dict[str, Any]) -> list[dict[str, str]]:
        result = await market_cache.call_tool(session, name, arguments)
        if is_rate_limit(result):
            raise ValueError(RATE_LIMIT_RESPONSE)
        if result.isError:
            raise ValueError("The provider could not retrieve prices for this growth calculation.")
        text = "\n".join(item.text for item in result.content if item.type == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
        else:
            if isinstance(data, dict) and data.get("status") == "error":
                raise ValueError(
                    "The provider could not retrieve prices for this growth calculation."
                )
            rows = data.get("values", [data]) if isinstance(data, dict) else data
        if not rows or not isinstance(rows, list) or "close" not in rows[0]:
            raise ValueError("No usable closing prices were returned by the provider.")
        return rows

    async def history(asset: str, target: str) -> dict[str, dict[str, str]]:
        rows = await records(
            "get_time_series",
            {
                "symbol": asset,
                "interval": "1day",
                "outputsize": 15,
                "start_date": (date.fromisoformat(target) - timedelta(days=14)).isoformat(),
                "end_date": target,
            },
        )
        eligible = {
            row["datetime"][:10]: row
            for row in rows
            if row.get("datetime", "") and row["datetime"][:10] <= target
        }
        if not eligible:
            raise ValueError("No closing price on or before the requested date within 14 days.")
        return eligible

    try:
        endpoint = parameters["end_date"]
        # Comparisons use completed daily bars to avoid mixing live and closing quotes.
        if comparison and endpoint == as_of:
            endpoint = (date.fromisoformat(as_of) - timedelta(days=1)).isoformat()
        if not comparison and endpoint == as_of:
            # Different endpoints, with no dependency between their requests.
            start_rows, quote_rows = await asyncio.gather(
                history(symbols[0], parameters["start_date"]),
                records("get_quote", {"symbol": symbols[0]}),
            )
            starts = [start_rows]
            row = quote_rows[0]
            ends = [{row["datetime"][:10]: row}]
            note = "Endpoint is the latest available quote and may be intraday or delayed."
        else:
            # Calls to the same API are sequential, including across comparison assets.
            starts = [await history(asset, parameters["start_date"]) for asset in symbols]
            ends = [await history(asset, endpoint) for asset in symbols]
            note = "Endpoints use daily closes; comparisons use common observation dates."
        common_start = set(starts[0]).intersection(*(set(rows) for rows in starts[1:]))
        common_end = set(ends[0]).intersection(*(set(rows) for rows in ends[1:]))
        if not common_start or not common_end:
            raise ValueError("No common observation dates within the 14-day lookup windows.")
        start_date, end_date = max(common_start), max(common_end)
        if end_date < start_date:
            raise ValueError("Provider observation dates are out of order.")
        calculations = []
        for asset, start_rows, end_rows in zip(symbols, starts, ends, strict=True):
            start, end = start_rows[start_date], end_rows[end_date]
            calculation = compute_price_growth(start, end)
            calculation.update(
                {
                    "requested_start_date": parameters["start_date"],
                    "requested_end_date": parameters["end_date"],
                    "symbol": asset,
                    "currency": end.get("currency", start.get("currency", "unavailable")),
                }
            )
            calculations.append(calculation)
        if comparison:
            baseline = Decimal(calculations[0]["growth_percent"])
            lines = [
                f"Price-growth comparison: requested {parameters['start_date']} to "
                f"{parameters['end_date']}; aligned observations {start_date} to {end_date}.",
                "",
                "| Asset | Currency | Start | End | Price change | Growth | "
                f"Delta vs {symbols[0]} | Direction |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for item in calculations:
                growth = Decimal(item["growth_percent"])
                delta = growth - baseline
                item["relative_delta_pp"] = str(delta)
                item["direction"] = "up" if growth > 0 else "down" if growth < 0 else "flat"
                lines.append(
                    f"| {item['symbol']} | {item['currency']} | {item['start_price']} | "
                    f"{item['end_price']} | {item['price_change']} | {growth:.2f}% | "
                    f"{delta:+.2f} pp | {item['direction']} |"
                )
            lines.extend(
                [
                    "",
                    *(f"{item['symbol']}: {item['math']}" for item in calculations),
                    "",
                    note + " Deltas are percentage-point differences in local-currency "
                    "price growth, excluding dividends and FX effects. Direction summarizes "
                    "endpoint movement; volatility and intra-period trends are not evaluated.",
                ]
            )
            return {"answer": "\n".join(lines), "calculations": calculations, "status": "answered"}
        item = calculations[0]
        answer = (
            f"{symbols[0]} price growth from {parameters['start_date']} to "
            f"{parameters['end_date']} (currency: {item['currency']}):\n"
            f"Starting close ({item['start_date']}): {item['start_price']}\n"
            f"Ending price ({item['end_date']}): {item['end_price']}\n"
            f"Price change: {item['end_price']} - {item['start_price']} = {item['price_change']}\n"
            f"Growth: {item['math']}\n{note} Baseline uses the closest available close on or "
            "before the start date. This is price growth, excluding dividends."
        )
        return {"answer": answer, "calculation": item, "status": "answered"}
    except (ValueError, KeyError, InvalidOperation) as error:
        return {"answer": str(error), "status": "error"}
