import json
from unittest.mock import AsyncMock, Mock

from langsmith import Client, tracing_context
from langsmith.run_trees import RunTree
from mcp.types import CallToolResult, TextContent

from finassist.agents.market_analysis.cache import MarketCache
from finassist.agents.market_analysis.tracing import tool_inputs, tool_output


def result(payload):
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


async def test_mcp_attempts_are_children_of_request_cache_span():
    client = Mock(spec=Client)
    parent = RunTree(name="request", run_type="chain", client=client)
    session = AsyncMock()
    session.call_tool.return_value = result({"symbol": "AAPL", "close": "100"})
    cache = MarketCache()
    with tracing_context(parent=parent, enabled=True, client=client):
        first = await cache.call_tool(session, "get_quote", {"symbol": "AAPL"})
        second = await cache.call_tool(session, "get_quote", {"symbol": "AAPL"})
    assert first is second
    assert len(parent.child_runs) == 2
    miss, hit = parent.child_runs
    assert miss.name == hit.name == "Market data cache"
    assert miss.extra["metadata"]["cache_status"] == "miss"
    assert hit.extra["metadata"]["cache_status"] == "hit"
    assert len(miss.child_runs) == 1
    assert not hit.child_runs
    attempt = miss.child_runs[0]
    assert attempt.name == "Call market MCP tool"
    assert attempt.trace_id == parent.trace_id
    assert attempt.inputs == {"tool": "get_quote", "arguments": {"symbol": "AAPL"}}
    assert attempt.outputs == {"isError": False, "record_count": 1}
    assert "session" not in attempt.inputs


async def test_provider_error_has_safe_details_and_bypass_metadata():
    client = Mock(spec=Client)
    parent = RunTree(name="request", run_type="chain", client=client)
    session = AsyncMock()
    session.call_tool.return_value = result(
        {
            "status": "error",
            "code": 404,
            "message": "Not found https://provider.test?apikey=private token=private",
        }
    )
    cache = MarketCache()
    with tracing_context(parent=parent, enabled=True, client=client):
        await cache.call_tool(session, "get_quote", {"symbol": "AAP", "api_key": "private"})
    run = parent.child_runs[0]
    assert run.extra["metadata"]["cache_status"] == "bypass_provider_error"
    assert run.child_runs[0].outputs["code"] == "404"
    assert "private" not in json.dumps(run.inputs)
    assert "private" not in json.dumps(run.child_runs[0].outputs)
    assert not cache.entries


def test_safe_selectors_and_no_full_quote_output():
    assert tool_inputs(
        {
            "session": "secret",
            "self": "secret",
            "name": "get_quote",
            "arguments": {"symbol": "AAP", "api_key": "secret"},
        }
    ) == {
        "tool": "get_quote",
        "arguments": {"symbol": "AAP"},
    }
    assert "100" not in json.dumps(tool_output(result({"close": "100"})))


async def test_transport_exception_trace_excludes_provider_credentials():
    import pytest

    client = Mock(spec=Client)
    parent = RunTree(name="request", run_type="chain", client=client)
    session = AsyncMock()
    session.call_tool.side_effect = ValueError("private-api-key https://example.test")
    with tracing_context(parent=parent, enabled=True, client=client):
        with pytest.raises(RuntimeError, match="ValueError"):
            await MarketCache().call_tool(session, "get_quote", {"symbol": "AAP"})
    error = parent.child_runs[0].child_runs[0].error
    assert "private-api-key" not in error
    assert "https://example.test" not in error


async def test_reused_connection_traces_follow_each_request():
    from contextlib import asynccontextmanager

    from finassist.agents.market_analysis.session import MarketSessionManager
    from finassist.core.config import Settings

    client = Mock(spec=Client)
    raw = AsyncMock()
    raw.call_tool.return_value = result({"close": "100"})

    @asynccontextmanager
    async def connect(settings):
        yield raw

    manager = MarketSessionManager(Settings(_env_file=None), connect)
    parents = [RunTree(name=f"request-{i}", run_type="chain", client=client) for i in range(2)]
    try:
        for parent, symbol in zip(parents, ["AAP", "AAPL"], strict=True):
            with tracing_context(parent=parent, enabled=True, client=client):
                session = await manager.get_session()
                await MarketCache().call_tool(session, "get_quote", {"symbol": symbol})
        for parent in parents:
            assert parent.child_runs[0].child_runs[0].trace_id == parent.trace_id
        assert parents[0].trace_id != parents[1].trace_id
    finally:
        await manager.aclose()
