"""Expose provider tool observations without deriving prices from model prose."""

import csv
import io
import json
from typing import Any


def provider_payload(result: Any) -> Any:
    """Decode MCP's optional wrapper once, without serializing duplicate text/data."""
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    if not data or isinstance(data, str):
        text = (
            data
            if isinstance(data, str)
            else "\n".join(item.text for item in result.content if item.type == "text")
        )
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = list(csv.DictReader(io.StringIO(text), delimiter=";")) or text
    return data


def chart_observation(result: Any, *, tool: str, symbol: str, interval: str) -> dict | None:
    if getattr(result, "isError", False):
        return None
    data = provider_payload(result)
    if isinstance(data, dict) and (data.get("status") == "error" or data.get("code")):
        return None
    if isinstance(data, dict):
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        rows = data.get("values", [data])
    else:
        meta, rows = {}, data
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    if not any(row.get("close") is not None for row in rows if isinstance(row, dict)):
        return None
    return {
        "type": "market_observations",
        "symbol": meta.get("symbol", rows[0].get("symbol", symbol)),
        "currency": meta.get("currency", rows[0].get("currency")),
        "exchange": meta.get("exchange", rows[0].get("exchange")),
        "provider": "Twelve Data",
        "interval": interval if tool == "get_time_series" else "quote",
        "values": rows,
    }
