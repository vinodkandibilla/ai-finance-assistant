"""Tavily news retrieval; evidence IDs come from checkpointed tool history."""

from datetime import date, timedelta
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import ValidationError

from finassist.agents.news_synthesizer.schemas import NewsEvidence
from finassist.core.config import Settings, get_settings


def create_news_search_tool(settings: Settings | None = None):
    """Bind settings; use sequential search calls so history assigns unique execution IDs."""
    settings = settings or get_settings()

    @tool("search_financial_news", response_format="content_and_artifact")
    async def search_financial_news(
        query: str,
        state: Annotated[dict, InjectedState],
        time_range: Literal["day", "week", "month", "year"] = "week",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[dict, dict]:
        """Retrieve up to three news excerpts for a requested timeframe using Tavily.

        time_range is a rolling day/week/month/year (default week). For calendar periods
        such as today or last calendar month, supply both ISO YYYY-MM-DD dates;
        explicit dates are inclusive calendar dates and override time_range.
        Python converts the end date to Tavily's exclusive upper bound.
        Returns evidence IDs, exact source URLs, available publication dates and content.
        Retrieval only: no final answer or synthesis. Search calls must run sequentially.
        """

        def respond(status, evidence=None, error=None):
            payload = {
                "status": status,
                "evidence": evidence or [],
                "errors": [error] if error else [],
            }
            return payload, payload

        if not query.strip():
            return respond("error", error="invalid_query")
        timeframe = {"time_range": time_range}
        if start_date is not None or end_date is not None:
            try:
                if (
                    not start_date
                    or not end_date
                    or (date.fromisoformat(start_date) > date.fromisoformat(end_date))
                ):
                    raise ValueError
                # Tavily searches before end_date and rejects equal endpoints.
                # Include the complete requested end day, including single-day searches.
                provider_end_date = (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()
            except (ValueError, OverflowError):
                return respond("error", error="invalid_news_timeframe")
            timeframe = {"start_date": start_date, "end_date": provider_end_date}
        if settings.tavily_api_key is None:
            return respond("error", error="tavily_not_configured")
        try:
            async with httpx.AsyncClient(timeout=settings.tavily_timeout_seconds) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={
                        "Authorization": "Bearer " + settings.tavily_api_key.get_secret_value()
                    },
                    json={
                        "query": query,
                        "topic": "news",
                        **timeframe,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError:
            return respond("error", error="news_provider_error")
        try:
            body = response.json()
        except ValueError:
            return respond("error", error="malformed_news_results")
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            return respond("error", error="malformed_news_results")
        if not results:
            return respond("insufficient", error="no_news_results")

        # No global counter: threads start at E1 and checkpointed searches keep their IDs.
        known = {}
        for message in state.get("messages", []):
            if isinstance(message, ToolMessage) and message.name == "search_financial_news":
                for item in (message.artifact or {}).get("evidence", []):
                    known[item["url"]] = item["evidence_id"]
        next_id = max((int(value[1:]) for value in known.values()), default=0) + 1
        evidence = []
        for item in results[:3]:
            if not isinstance(item, dict):
                continue
            try:
                url = item.get("url")
                identifier = known.get(url, f"E{next_id}") if isinstance(url, str) else "E1"
                document = NewsEvidence(
                    evidence_id=identifier,
                    title=item.get("title"),
                    url=url,
                    source=item.get("source")
                    or (urlsplit(url).hostname if isinstance(url, str) else None),
                    published_at=item.get("published_date"),
                    content=item.get("content"),
                )
            except (ValidationError, ValueError):
                continue
            if document.url not in known:
                known[document.url] = document.evidence_id
                next_id += 1
            evidence.append(document.model_dump())
        if not evidence:
            return respond("error", error="malformed_news_results")
        return respond("success", evidence)

    return search_financial_news
