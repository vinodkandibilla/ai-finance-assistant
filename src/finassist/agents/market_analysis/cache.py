"""Shared, process-local TTL/LRU cache for market MCP responses."""

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from typing import Any

from langsmith import traceable
from mcp import ClientSession
from mcp.types import CallToolResult

from finassist.agents.market_analysis.tracing import (
    cache_status,
    invoke_mcp_tool,
    tool_inputs,
    tool_output,
)
from finassist.core.config import get_settings

TTLS = {"get_quote": 30 * 60, "search_symbol": 24 * 60 * 60, "get_time_series": 24 * 60 * 60}
logger = logging.getLogger(__name__)
_request_calls: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "market_request_calls", default=None
)


@asynccontextmanager
async def market_request_budget(limit: int) -> AsyncIterator[None]:
    token = _request_calls.set((0, limit))
    try:
        yield
    finally:
        _request_calls.reset(token)


class MarketCache:
    def __init__(
        self,
        capacity: int = 100,
        *,
        max_concurrency: int = 2,
        max_retries: int = 0,
        backoff_base: float = 0.5,
        backoff_max: float = 4.0,
    ) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[str, tuple[float, CallToolResult]] = OrderedDict()
        self.pending: dict[str, asyncio.Task[CallToolResult]] = {}
        self.limiter = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

    @traceable(
        name="Market data cache",
        run_type="tool",
        process_inputs=tool_inputs,
        process_outputs=tool_output,
    )
    async def call_tool(
        self, session: ClientSession, name: str, arguments: dict[str, Any], *, retry: bool = True
    ) -> CallToolResult:
        if name not in TTLS:
            cache_status("uncached_tool")
            return await invoke_mcp_tool(session, name, arguments)
        normalized = dict(arguments)
        if "symbol" in normalized:
            normalized["symbol"] = normalized["symbol"].strip()
            if name != "search_symbol":
                normalized["symbol"] = normalized["symbol"].upper()
        key = name + ":" + json.dumps(normalized, sort_keys=True)
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        # Only public provider selectors are logged; never dump arbitrary tool args.
        selectors = {
            field: normalized[field]
            for field in (
                "start_date",
                "end_date",
                "outputsize",
                "exchange",
                "country",
                "currency",
                "timezone",
                "order",
                "format",
                "adjust",
            )
            if field in normalized
        }
        now = monotonic()
        expired_exact = key in self.entries and self.entries[key][0] <= now
        for expired in [key for key, (expiry, _) in self.entries.items() if expiry <= now]:
            del self.entries[expired]
        if key in self.entries:
            self.entries.move_to_end(key)
            cache_status("hit")
            logger.info(
                "[Market Cache] HIT | tool=%s | symbol=%r | interval=%r | "
                "remaining_ttl_seconds=%.0f | pid=%s | key=%s | selectors=%s",
                name,
                normalized.get("symbol"),
                normalized.get("interval"),
                self.entries[key][0] - now,
                os.getpid(),
                fingerprint,
                json.dumps(selectors, sort_keys=True),
            )
            return self.entries[key][1]

        reason = (
            "expired"
            if expired_exact
            else ("empty_cache" if not self.entries else "key_not_present")
        )
        logger.info(
            "[Market Cache] MISS | tool=%s | symbol=%r | interval=%r | shared_inflight=%s "
            "| pid=%s | key=%s | reason=%s | entries=%s | selectors=%s",
            name,
            normalized.get("symbol"),
            normalized.get("interval"),
            key in self.pending,
            os.getpid(),
            fingerprint,
            reason,
            len(self.entries),
            json.dumps(selectors, sort_keys=True),
        )

        cache_status("shared_inflight" if key in self.pending else "miss")

        async def fetch() -> CallToolResult:
            state = _request_calls.get()
            if state is not None:
                calls, limit = state
                if calls >= limit:
                    raise RuntimeError("Market provider request budget exhausted.")
                _request_calls.set((calls + 1, limit))
            retries = self.max_retries if retry else 0
            for attempt in range(retries + 1):
                try:
                    async with self.limiter:
                        result = await invoke_mcp_tool(session, name, normalized)
                except Exception:
                    if attempt >= retries:
                        raise
                    await asyncio.sleep(min(self.backoff_max, self.backoff_base * (2**attempt)))
                    continue
                text = "\n".join(block.text for block in result.content if block.type == "text")
                retryable = result.isError or bool(
                    re.search(r"\b429\b|rate[ _-]?limit|too many requests|temporar", text, re.I)
                )
                if retryable and attempt < retries:
                    await asyncio.sleep(min(self.backoff_max, self.backoff_base * (2**attempt)))
                    continue
                break
            # MCP providers can report errors as text with isError=False.
            if not result.isError and not re.search(
                r"\berror\b|\b429\b|rate[ _-]?limit|too many requests|api credits", text, re.I
            ):
                self.entries[key] = (monotonic() + TTLS[name], result)
                self.entries.move_to_end(key)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
            else:
                cache_status("bypass_provider_error")
                logger.info(
                    "[Market Cache] BYPASS | tool=%s | key=%s | reason=provider_error",
                    name,
                    fingerprint,
                )
            return result

        if key not in self.pending:
            self.pending[key] = asyncio.create_task(fetch())
        try:
            return await self.pending[key]
        finally:
            self.pending.pop(key, None)


_settings = get_settings()
market_cache = MarketCache(
    capacity=_settings.market_cache_max_entries,
    max_concurrency=_settings.market_provider_max_concurrency,
    max_retries=_settings.market_provider_max_retries,
    backoff_base=_settings.market_provider_backoff_base_seconds,
    backoff_max=_settings.market_provider_backoff_max_seconds,
)
