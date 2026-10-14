import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from finassist.agents.market_analysis import cache as module
from finassist.agents.market_analysis.cache import TTLS, MarketCache


def result(text="datetime;close\n2025-01-01;100", error=False):
    return CallToolResult(isError=error, content=[TextContent(type="text", text=text)])


@pytest.mark.parametrize("name", list(TTLS))
async def test_fixed_ttl_and_shared_response(name, monkeypatch):
    clock = [0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    cache = MarketCache()
    first, second = AsyncMock(), AsyncMock()
    first.call_tool.return_value = result()
    second.call_tool.return_value = result()
    args = {"symbol": "AAPL"}
    await cache.call_tool(first, name, args)
    clock[0] = TTLS[name] - 1
    await cache.call_tool(second, name, args)
    second.call_tool.assert_not_awaited()
    clock[0] = TTLS[name]
    await cache.call_tool(second, name, args)
    second.call_tool.assert_awaited_once()


async def test_lru_eviction():
    cache, session = MarketCache(capacity=2), AsyncMock()
    session.call_tool.return_value = result()
    for symbol in ["AAPL", "MSFT", "AAPL", "META"]:
        await cache.call_tool(session, "get_quote", {"symbol": symbol})
    assert len(cache.entries) == 2
    await cache.call_tool(session, "get_quote", {"symbol": "MSFT"})
    assert session.call_tool.await_count == 4


async def test_identical_requests_share_one_fetch():
    cache, session = MarketCache(), AsyncMock()

    async def fetch(name, arguments):
        await asyncio.sleep(0)
        return result()

    session.call_tool.side_effect = fetch
    await asyncio.gather(
        *(cache.call_tool(session, "get_quote", {"symbol": " aapl "}) for _ in range(3))
    )
    session.call_tool.assert_awaited_once_with("get_quote", {"symbol": "AAPL"})


@pytest.mark.parametrize(
    "response",
    [result(error=True), result("Error: unavailable"), result('{"code":429,"status":"error"}')],
)
async def test_errors_are_not_cached(response):
    cache, session = MarketCache(), AsyncMock()
    session.call_tool.return_value = response
    for _ in range(2):
        await cache.call_tool(session, "get_quote", {"symbol": "AAPL"})
    assert session.call_tool.await_count == 2
    assert not cache.entries
