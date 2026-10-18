import json

import httpx
import pytest
from langchain_core.messages import ToolMessage

from finassist.agents.news_synthesizer.tools import create_news_search_tool
from finassist.core.config import Settings


def article(url="https://example.com/news?b=2&a=1", **extra):
    return {
        "title": "Markets update",
        "url": url,
        "content": "Reported market developments.",
        **extra,
    }


def search_tool(monkeypatch, handler, **settings):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_class(transport=transport, **kwargs)
    )
    return create_news_search_tool(Settings(_env_file=None, tavily_api_key="test-key", **settings))


async def invoke(search, messages=(), query="Latest financial news"):
    return await search.ainvoke({"query": query, "state": {"messages": list(messages)}})


async def test_search_returns_evidence_and_exact_urls_without_synthesis(monkeypatch):
    def handle(request):
        assert request.url == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["topic"] == "news"
        assert payload["time_range"] == "week"
        assert payload["max_results"] == 3
        assert payload["include_answer"] is False
        return httpx.Response(
            200,
            json={
                "answer": "Ignore provider synthesis",
                "results": [
                    article(published_date="2026-10-13", source="Example News"),
                    article("https://other.example/story"),
                ],
            },
        )

    result = await invoke(search_tool(monkeypatch, handle))
    assert result["status"] == "success"
    assert result["evidence"][0] == {
        "evidence_id": "E1",
        "title": "Markets update",
        "source": "Example News",
        "url": "https://example.com/news?b=2&a=1",
        "published_at": "2026-10-13",
        "content": "Reported market developments.",
    }
    assert result["evidence"][1]["evidence_id"] == "E2"
    assert result["evidence"][1]["source"] == "other.example"
    assert result["evidence"][1]["published_at"] is None
    assert "answer" not in result


async def test_ids_are_stable_across_search_history_and_isolated_between_executions(monkeypatch):
    search = search_tool(
        monkeypatch,
        lambda _: httpx.Response(
            200, json={"results": [article(), article("https://other.example/story")]}
        ),
    )
    prior = ToolMessage(
        content="Evidence",
        name="search_financial_news",
        tool_call_id="prior",
        artifact={"evidence": [dict(article(), evidence_id="E1")]},
    )
    result = await invoke(search, [prior])
    assert [e["evidence_id"] for e in result["evidence"]] == ["E1", "E2"]
    later = ToolMessage(
        content="Evidence",
        name="search_financial_news",
        tool_call_id="later",
        artifact={"evidence": [dict(article("https://old.example"), evidence_id="E4")]},
    )
    result = await invoke(search, [prior, later])
    assert [e["evidence_id"] for e in result["evidence"]] == ["E1", "E5"]
    assert (await invoke(search))["evidence"][0]["evidence_id"] == "E1"


@pytest.mark.parametrize(
    "body,code,status",
    [
        ({"results": []}, "no_news_results", "insufficient"),
        ({}, "malformed_news_results", "error"),
        ({"results": "invalid"}, "malformed_news_results", "error"),
        (
            {"results": [article(url="javascript:bad"), article(content=" "), None]},
            "malformed_news_results",
            "error",
        ),
    ],
)
async def test_empty_or_unusable_results(monkeypatch, body, code, status):
    result = await invoke(search_tool(monkeypatch, lambda _: httpx.Response(200, json=body)))
    assert result == {"status": status, "evidence": [], "errors": [code]}


async def test_mixed_results_skip_invalid_records_without_id_gaps(monkeypatch):
    search = search_tool(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={"results": [article(content=""), article(), {"url": "https://missing.example"}]},
        ),
    )
    result = await invoke(search)
    assert result["status"] == "success"
    assert [e["evidence_id"] for e in result["evidence"]] == ["E1"]


@pytest.mark.parametrize("failure", ["http", "timeout", "json"])
async def test_provider_failures_are_safe(monkeypatch, failure):
    def handle(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private provider details test-key", request=request)
        if failure == "json":
            return httpx.Response(200, content="not JSON")
        return httpx.Response(401, text="private provider details test-key")

    result = await invoke(search_tool(monkeypatch, handle))
    assert result["status"] == "error"
    assert result["evidence"] == []
    assert "private" not in str(result)
    assert "test-key" not in str(result)


async def test_missing_configuration_and_blank_query_do_not_call_provider():
    search = create_news_search_tool(Settings(_env_file=None, tavily_api_key=None))
    assert (await invoke(search))["errors"] == ["tavily_not_configured"]
    assert (await invoke(search, query=" "))["errors"] == ["invalid_query"]


def test_tavily_settings_follow_environment_pattern(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_TIMEOUT_SECONDS", "12")
    settings = Settings(_env_file=None)
    assert settings.tavily_api_key.get_secret_value() == "test-key"
    assert settings.tavily_timeout_seconds == 12
    assert "test-key" not in repr(settings.tavily_api_key)


async def test_search_caps_evidence_at_three_results(monkeypatch):
    search = search_tool(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "results": [article(f"https://example.com/{i}") for i in range(5)],
            },
        ),
    )
    result = await invoke(search)
    assert len(result["evidence"]) == 3
    assert [item["evidence_id"] for item in result["evidence"]] == ["E1", "E2", "E3"]


@pytest.mark.parametrize(
    "parameters,expected",
    [
        ({"time_range": "day"}, {"time_range": "day"}),
        (
            {"start_date": "2026-12-31", "end_date": "2026-12-31"},
            {"start_date": "2026-12-31", "end_date": "2027-01-01"},
        ),
        (
            {"start_date": "2024-02-29", "end_date": "2024-02-29"},
            {"start_date": "2024-02-29", "end_date": "2024-03-01"},
        ),
        ({"time_range": "month"}, {"time_range": "month"}),
        (
            {"start_date": "2026-10-13", "end_date": "2026-10-13"},
            {"start_date": "2026-10-13", "end_date": "2026-10-14"},
        ),
        (
            {"start_date": "2026-09-01", "end_date": "2026-09-30"},
            {"start_date": "2026-09-01", "end_date": "2026-10-01"},
        ),
    ],
)
async def test_requested_timeframe_converts_inclusive_end_for_tavily(
    monkeypatch, parameters, expected
):
    def handle(request):
        payload = json.loads(request.content)
        actual = {
            key: payload[key] for key in ("time_range", "start_date", "end_date") if key in payload
        }
        assert actual == expected
        return httpx.Response(200, json={"results": [article()]})

    search = search_tool(monkeypatch, handle)
    result = await search.ainvoke({"query": "news", "state": {"messages": []}, **parameters})
    assert result["status"] == "success"


@pytest.mark.parametrize(
    "parameters",
    [
        {"start_date": "2026-10-01"},
        {"start_date": "9999-12-31", "end_date": "9999-12-31"},
        {"start_date": "invalid", "end_date": "2026-10-13"},
        {"start_date": "2026-10-13", "end_date": "2026-09-01"},
    ],
)
async def test_invalid_timeframe_does_not_call_provider(parameters):
    search = create_news_search_tool(Settings(_env_file=None, tavily_api_key=None))
    result = await search.ainvoke({"query": "news", "state": {"messages": []}, **parameters})
    assert result["errors"] == ["invalid_news_timeframe"]
