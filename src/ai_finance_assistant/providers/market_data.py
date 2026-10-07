from __future__ import annotations

import asyncio
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
import structlog
from langchain_core.tools import tool

from ai_finance_assistant.core.config import get_settings

logger = structlog.get_logger(__name__)

# Compatibility defaults for legacy module-level access; runtime reads come from Settings.
ALPHA_VANTAGE_GLOBAL_QUOTE_URL = get_settings().alpha_vantage_base_url
QUOTE_CACHE_TTL_SECONDS = get_settings().quote_cache_ttl_seconds
SUPPORTED_SYMBOLS = frozenset({"AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "NVDA", "TSLA"})
COMPANY_NAME_TO_SYMBOL = {
    "AAPL": "AAPL",
    "APPLE": "AAPL",
    "MSFT": "MSFT",
    "MICROSOFT": "MSFT",
    "AMZN": "AMZN",
    "AMAZON": "AMZN",
    "GOOG": "GOOG",
    "GOOGLE": "GOOG",
    "ALPHABET": "GOOG",
    "GOOGL": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "NVDA": "NVDA",
    "NVIDIA": "NVDA",
    "TSLA": "TSLA",
    "TESLA": "TSLA",
}
QUOTE_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "market_quotes.json"

_symbol_locks: dict[str, asyncio.Lock] = {}


def extract_supported_symbol(query: str) -> str | None:
    matches = re.findall(r"(?<![A-Z0-9])[A-Z]{1,10}(?![A-Z0-9])", query.upper())
    for match in matches:
        if match in SUPPORTED_SYMBOLS:
            return match
        normalized_match = COMPANY_NAME_TO_SYMBOL.get(match)
        if normalized_match is not None:
            return normalized_match
    return None


class DiskCache:
    def __init__(self, path: Path = QUOTE_CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, dict[str, float | str]]:
        if not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

        if not isinstance(payload, dict):
            return {}

        return {
            str(symbol): values
            for symbol, values in payload.items()
            if isinstance(values, dict)
        }

    def _write(self, data: dict[str, dict[str, float | str]]) -> None:
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True)
        temp_path.replace(self.path)

    def get(self, symbol: str, *, now: float | None = None) -> str | None:
        current_time = time.time() if now is None else now
        entries = self._read()
        entry = entries.get(symbol)
        if entry is None:
            return None

        ttl_seconds = get_settings().quote_cache_ttl_seconds
        fetched_at = float(entry.get("fetched_at", 0.0))
        if current_time - fetched_at >= ttl_seconds:
            entries.pop(symbol, None)
            self._write(entries)
            return None

        price = entry.get("price")
        return str(price) if price is not None else None

    def set(self, symbol: str, price: str, *, fetched_at: float | None = None) -> None:
        timestamp = time.time() if fetched_at is None else fetched_at
        entries = self._read()
        entries[symbol] = {"price": str(price), "fetched_at": float(timestamp)}
        self._write(entries)


@tool
async def get_stock_price(symbol: str) -> str:
    """Get the latest Alpha Vantage quote price for a supported MAG7 stock symbol."""
    normalized_symbol = symbol.strip().upper()
    if normalized_symbol not in SUPPORTED_SYMBOLS:
        unsupported_symbol = normalized_symbol or "No symbol"
        return f"Stock price unavailable: {unsupported_symbol} is not a supported MAG7 ticker."

    cache = DiskCache(QUOTE_CACHE_PATH)
    cached_price = cache.get(normalized_symbol)
    if cached_price is not None:
        return cached_price

    lock = _symbol_locks.setdefault(normalized_symbol, asyncio.Lock())
    async with lock:
        cached_price = cache.get(normalized_symbol)
        if cached_price is not None:
            return cached_price

        settings = get_settings()
        api_key = (
            settings.alpha_vantage_api_key.get_secret_value()
            if settings.alpha_vantage_api_key
            else None
        )
        if not api_key:
            return (
                f"Stock price unavailable for {normalized_symbol}: "
                "Alpha Vantage API key is not configured."
            )

        settings = get_settings()
        alpha_vantage_url = settings.alpha_vantage_base_url
        timeout_seconds = settings.alpha_vantage_timeout_seconds

        try:
            logger.info(
                "invoking_alpha_vantage_quote",
                symbol=normalized_symbol,
                endpoint=alpha_vantage_url,
            )
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.get(
                    alpha_vantage_url,
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": normalized_symbol,
                        "apikey": api_key,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return f"Stock price unavailable for {normalized_symbol}: Alpha Vantage request failed."

        quote = payload.get("Global Quote") if isinstance(payload, dict) else None
        raw_price = quote.get("05. price") if isinstance(quote, dict) else None
        if not isinstance(raw_price, str) or not raw_price.strip():
            return f"Stock price unavailable for {normalized_symbol}: no quote was returned."

        try:
            parsed_price = Decimal(raw_price.strip())
        except InvalidOperation:
            return f"Stock price unavailable for {normalized_symbol}: the quote price was invalid."
        if not parsed_price.is_finite() or parsed_price < 0:
            return f"Stock price unavailable for {normalized_symbol}: the quote price was invalid."

        price = raw_price.strip()
        cache.set(normalized_symbol, price)
        return price
