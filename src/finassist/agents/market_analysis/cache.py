"""Shared, process-local TTL/LRU cache for market MCP responses."""

import asyncio
import json
import re
from collections import OrderedDict
from time import monotonic
from typing import Any

from mcp import ClientSession
from mcp.types import CallToolResult

from finassist.core.config import get_settings

TTLS = {"get_quote": 30 * 60, "search_symbol": 24 * 60 * 60, "get_time_series": 24 * 60 * 60}


class MarketCache:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[str, tuple[float, CallToolResult]] = OrderedDict()
        self.pending: dict[str, asyncio.Task[CallToolResult]] = {}

    async def call_tool(
        self, session: ClientSession, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        if name not in TTLS:
            return await session.call_tool(name, arguments)
        normalized = dict(arguments)
        if "symbol" in normalized:
            normalized["symbol"] = normalized["symbol"].strip()
            if name != "search_symbol":
                normalized["symbol"] = normalized["symbol"].upper()
        key = name + ":" + json.dumps(normalized, sort_keys=True)
        now = monotonic()
        for expired in [key for key, (expiry, _) in self.entries.items() if expiry <= now]:
            del self.entries[expired]
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][1]

        async def fetch() -> CallToolResult:
            result = await session.call_tool(name, normalized)
            text = "\n".join(block.text for block in result.content if block.type == "text")
            # MCP providers can report errors as text with isError=False.
            if not result.isError and not re.search(
                r"\berror\b|\b429\b|rate[ _-]?limit|too many requests|api credits", text, re.I
            ):
                self.entries[key] = (monotonic() + TTLS[name], result)
                self.entries.move_to_end(key)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
            return result

        if key not in self.pending:
            self.pending[key] = asyncio.create_task(fetch())
        try:
            return await self.pending[key]
        finally:
            self.pending.pop(key, None)


market_cache = MarketCache(capacity=get_settings().market_cache_max_entries)
