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


async def test_hit_logs_metadata_but_not_response_or_unrelated_arguments(caplog):
    import logging

    cache, session = MarketCache(), AsyncMock()
    session.call_tool.return_value = result("private-response-sentinel")
    arguments = {"symbol": "aapl", "interval": "1day", "unused": "sensitive-sentinel"}
    with caplog.at_level(logging.INFO, logger=module.__name__):
        await cache.call_tool(session, "get_time_series", arguments)
        assert "[Market Cache] HIT" not in caplog.text
        assert "[Market Cache] MISS" in caplog.text
        await cache.call_tool(session, "get_time_series", arguments)
    assert "[Market Cache] HIT" in caplog.text
    assert "tool=get_time_series" in caplog.text
    assert "symbol='AAPL'" in caplog.text
    assert "interval='1day'" in caplog.text
    assert "remaining_ttl_seconds=" in caplog.text
    assert "private-response-sentinel" not in caplog.text
    assert "sensitive-sentinel" not in caplog.text
    session.call_tool.assert_awaited_once()


async def test_miss_diagnostics_distinguish_selectors_and_expiration(caplog, monkeypatch):
    import logging

    clock = [0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    cache, session = MarketCache(), AsyncMock()
    session.call_tool.return_value = result()
    with caplog.at_level(logging.INFO, logger=module.__name__):
        await cache.call_tool(session, "get_time_series", {"symbol": "AAPL", "outputsize": 22})
        assert "reason=empty_cache" in caplog.text
        caplog.clear()
        await cache.call_tool(session, "get_time_series", {"symbol": "AAPL", "outputsize": 23})
        assert "reason=key_not_present" in caplog.text
        assert '"outputsize": 23' in caplog.text
        assert "pid=" in caplog.text and "key=" in caplog.text
        caplog.clear()
        clock[0] = TTLS["get_time_series"]
        await cache.call_tool(session, "get_time_series", {"symbol": "AAPL", "outputsize": 23})
        assert "reason=expired" in caplog.text


@pytest.mark.parametrize("text", [
    '{"code":503,"status":"error"}', 'HTTP 502 Bad Gateway',
    'HTTP 500 Internal Server Error', 'HTTP 504 Gateway Timeout',
    '429 Too many requests', 'Service temporarily unavailable',
])
async def test_transient_responses_retry_then_cache_success(text):
    cache, session = MarketCache(max_retries=2, backoff_base=0), AsyncMock()
    session.call_tool.side_effect = [result(text, error=True), result()]
    await cache.call_tool(session, "get_quote", {"symbol": "AAPL"})
    assert session.call_tool.await_count == 2
    assert len(cache.entries) == 1


@pytest.mark.parametrize("text", [
    'HTTP 400 Bad Request', 'HTTP 401 Unauthorized', 'HTTP 403 Forbidden',
    'HTTP 404 Not Found', 'HTTP 501 Not Implemented', 'HTTP 505 unsupported version',
    'Invalid symbol', 'Invalid API key', 'API credits exhausted', 'Unknown error',
    '{"code":400,"message":"temporarily invalid parameter"}',
])
async def test_permanent_and_unknown_errors_do_not_retry(text):
    cache, session = MarketCache(max_retries=2, backoff_base=0), AsyncMock()
    session.call_tool.return_value = result(text, error=True)
    await cache.call_tool(session, "get_quote", {"symbol": "AAPL"})
    assert session.call_tool.await_count == 1
    assert not cache.entries


@pytest.mark.parametrize("error, attempts", [
    (TimeoutError("timeout"), 3), (ConnectionResetError("reset"), 3),
    (RuntimeError("HTTP 503 Service Unavailable"), 3),
    (ValueError("invalid input"), 1), (RuntimeError("HTTP 401"), 1),
])
async def test_exception_retry_classification_survives_trace_sanitization(error, attempts):
    cache, session = MarketCache(max_retries=2, backoff_base=0), AsyncMock()
    session.call_tool.side_effect = error
    with pytest.raises((RuntimeError, TimeoutError)):
        await cache.call_tool(session, "get_quote", {"symbol": "AAPL"})
    assert session.call_tool.await_count == attempts
    assert not cache.entries
    assert not cache.pending


async def test_retry_false_disables_transient_retry():
    cache, session = MarketCache(max_retries=2, backoff_base=0), AsyncMock()
    session.call_tool.return_value = result('HTTP 503', error=True)
    await cache.call_tool(session, "get_quote", {"symbol": "AAPL"}, retry=False)
    session.call_tool.assert_awaited_once()
