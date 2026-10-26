import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from mcp.types import ListToolsResult, Tool
from pydantic import ValidationError

from finassist.agents.market_analysis import session as module
from finassist.agents.market_analysis.session import MarketSessionManager
from finassist.core.config import Settings


def manager(*, ttl=1800):
    connections, owners, exits = [], [], []

    @asynccontextmanager
    async def connect(settings):
        owner = asyncio.current_task()
        session = AsyncMock()
        session.list_tools.return_value = ListToolsResult(
            tools=[Tool(name="get_quote", inputSchema={"type": "object"})]
        )
        connections.append(session)
        owners.append(owner)
        try:
            yield session
        finally:
            assert asyncio.current_task() is owner
            exits.append(owner)

    pool = MarketSessionManager(
        Settings(_env_file=None, market_tool_catalog_ttl_seconds=ttl), connect
    )
    return pool, connections, owners, exits


async def test_concurrent_reuse_catalog_and_owner_shutdown():
    pool, connections, owners, exits = manager()
    try:
        first, second = await asyncio.gather(pool.get_session(), pool.get_session())
        assert first is second
        assert len(connections) == 1
        catalogs = await asyncio.gather(first.list_tools(), second.list_tools())
        connections[0].list_tools.assert_awaited_once()
        catalogs[0].tools.clear()
        assert (await second.list_tools()).tools  # Consumers cannot mutate the cache.
        assert owners[0] is not asyncio.current_task()
    finally:
        await pool.aclose()
    assert exits == owners
    await pool.aclose()  # Idempotent.
    with pytest.raises(ConnectionError):
        await pool.get_session()


async def test_catalog_ttl_refresh(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    pool, connections, _, _ = manager(ttl=1800)
    try:
        session = await pool.get_session()
        await session.list_tools()
        clock[0] += 1799
        await session.list_tools()
        assert connections[0].list_tools.await_count == 1
        clock[0] += 1
        await session.list_tools()
        assert connections[0].list_tools.await_count == 2
    finally:
        await pool.aclose()


async def test_transport_error_reconnects_and_refreshes_catalog():
    pool, connections, _, exits = manager()
    try:
        first = await pool.get_session()
        await first.list_tools()
        connections[0].call_tool.side_effect = ConnectionError("broken pipe")
        with pytest.raises(ConnectionError):
            await first.call_tool("get_quote", {"symbol": "AAPL"})
        second = await pool.get_session()
        assert second is not first
        assert len(connections) == 2
        assert len(exits) == 1
        await second.list_tools()
        connections[1].list_tools.assert_awaited_once()
        assert connections[0].call_tool.await_count == 1  # Failed request not replayed.
    finally:
        await pool.aclose()


async def test_catalog_failure_reconnects():
    pool, connections, _, _ = manager()
    try:
        session = await pool.get_session()
        connections[0].list_tools.side_effect = ConnectionError()
        with pytest.raises(ConnectionError):
            await session.list_tools()
        assert await pool.get_session() is not session
    finally:
        await pool.aclose()


async def test_request_cancellation_does_not_close_shared_session():
    pool, connections, _, exits = manager()
    started = asyncio.Event()

    async def call(name, args):
        if args["symbol"] == "AAPL":
            started.set()
            await asyncio.Event().wait()
        return "MSFT result"

    try:
        session = await pool.get_session()
        connections[0].call_tool.side_effect = call
        task = asyncio.create_task(session.call_tool("get_quote", {"symbol": "AAPL"}))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await session.call_tool("get_quote", {"symbol": "MSFT"}) == "MSFT result"
        assert await pool.get_session() is session
        assert not exits
    finally:
        await pool.aclose()


async def test_cancelled_startup_waiter_does_not_cancel_owner():
    entered, allow = asyncio.Event(), asyncio.Event()
    exits = []
    raw = AsyncMock()

    @asynccontextmanager
    async def connect(settings):
        owner = asyncio.current_task()
        entered.set()
        await allow.wait()
        try:
            yield raw
        finally:
            assert asyncio.current_task() is owner
            exits.append(True)

    pool = MarketSessionManager(Settings(_env_file=None), connect)
    try:
        waiter = asyncio.create_task(pool.get_session())
        await asyncio.wait_for(entered.wait(), 1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        allow.set()
        assert await pool.get_session()
    finally:
        await pool.aclose()
    assert exits == [True]


async def test_failed_startup_can_retry_without_leaking_credentials():
    calls = 0

    @asynccontextmanager
    async def connect(settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private key")
        yield AsyncMock()

    pool = MarketSessionManager(Settings(_env_file=None), connect)
    try:
        with pytest.raises(ConnectionError, match="unavailable") as error:
            await pool.get_session()
        assert "private key" not in str(error.value)
        assert await pool.get_session()
        assert calls == 2
    finally:
        await pool.aclose()


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan")])
def test_invalid_catalog_ttl_rejected(ttl):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, market_tool_catalog_ttl_seconds=ttl)


async def test_agent_uses_shared_session_for_repeated_quotes(monkeypatch):
    from langchain_core.messages import AIMessage
    from mcp.types import CallToolResult, TextContent

    from test_market_analysis_agent import make_agent

    agent, raw, _ = make_agent(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "first"}],
            ),
            AIMessage(content="AAPL quote"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "second"}],
            ),
            AIMessage(content="Cached AAPL quote"),
        ],
    )
    connections = []

    @asynccontextmanager
    async def connect(settings):
        connections.append(True)
        yield raw

    raw.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text='{"close":"100"}')]
    )
    pool = MarketSessionManager(agent.settings, connect)
    agent.sessions = pool
    try:
        for _ in range(2):
            assert (await agent.ainvoke({"query": "AAPL price"}))["status"] == "answered"
        assert connections == [True]
        raw.list_tools.assert_awaited_once()
        raw.call_tool.assert_awaited_once()
    finally:
        await pool.aclose()


def test_app_lifespan_closes_lazy_session_manager():
    from fastapi.testclient import TestClient

    from finassist.main import create_app

    app = create_app()
    with TestClient(app):
        pool = app.state.market_sessions
        assert pool.owner is None  # No subprocess for non-market traffic.
    assert pool.closed
    assert not hasattr(app.state, "market_sessions")


def test_catalog_ttl_environment_setting(monkeypatch):
    monkeypatch.setenv("MARKET_TOOL_CATALOG_TTL_SECONDS", "900")
    assert Settings(_env_file=None).market_tool_catalog_ttl_seconds == 900
