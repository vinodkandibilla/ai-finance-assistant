"""Compact model context and routine summaries; full observations remain for charts."""

import json
from typing import Any

from finassist.agents.market_analysis.chart_data import provider_payload


def compact_tool_result(result: Any) -> str:
    if getattr(result, "isError", False):
        return json.dumps(result.model_dump(mode="json"))
    data = provider_payload(result)
    # Encode repeated keys once, retaining every row/field in JSON or CSV results.
    rows = data.get("values") if isinstance(data, dict) else data
    if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        metadata = (
            {key: value for key, value in data.items() if key != "values"}
            if isinstance(data, dict)
            else {}
        )
        data = {
            **metadata,
            "columns": columns,
            "rows": [[row.get(key) for key in columns] for row in rows],
        }
    return json.dumps(data, separators=(",", ":"))


def market_response(answer: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    for observation in observations:
        quote = next(
            (
                item
                for item in observations
                if item["interval"] == "quote" and item["symbol"] == observation["symbol"]
            ),
            {},
        )
        for field in ("currency", "exchange"):
            if not observation.get(field) and quote.get(field):
                observation[field] = quote[field]
    return {
        "answer": answer,
        "status": "answered",
        **({"calculations": observations} if observations else {}),
    }


def format_snapshot(observations: list[dict[str, Any]]) -> str:
    """Describe provider data without deriving returns or inventing observation times."""
    lines = []
    for item in observations:
        symbol = item["symbol"]
        rows = item["values"]
        if item["interval"] == "quote":
            row = rows[0]
            currency = item.get("currency") or "currency unavailable"
            exchange = item.get("exchange") or "exchange unavailable"
            lines.append(
                f"{symbol} ({exchange}) — latest available price: {currency} {row['close']}."
            )
            details = []
            for field, label in (("change", "Change"), ("percent_change", "Change (%)")):
                if row.get(field) is not None:
                    details.append(f"{label}: {row[field]}")
            if details:
                lines.append("; ".join(details) + ".")
            observed = row.get("datetime") or row.get("timestamp")
            lines.append(f"Provider observation time: {observed or 'unavailable'}.")
            if row.get("is_market_open") is not None:
                lines.append(f"Provider market-open flag: {row['is_market_open']}.")
        else:
            dates = sorted(str(row["datetime"]) for row in rows if row.get("datetime"))
            window = f"{dates[0]} to {dates[-1]}" if dates else "dates unavailable"
            lines.append(
                f"{symbol} chart: {len(rows)} provider observations, "
                f"interval {item['interval']}, {window}."
            )
    lines.append("Source: Twelve Data; data may be cached. Current bars may be incomplete.")
    return "\n".join(lines)
