"""Quote and historical market agent backed by the official Twelve Data MCP server."""

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from finassist.agents.market_analysis.cache import market_cache, market_request_budget
from finassist.agents.market_analysis.chart_data import chart_observation
from finassist.agents.market_analysis.extrema import calculate_price_extrema
from finassist.agents.market_analysis.growth import (
    RATE_LIMIT_RESPONSE,
    calculate_price_growth,
    is_rate_limit,
)
from finassist.agents.market_analysis.presentation import (
    compact_tool_result,
    format_snapshot,
    market_response,
)
from finassist.agents.market_analysis.session import MarketSessionManager
from finassist.agents.market_analysis.timeframe import (
    MAX_RECORDS,
    resolve_market_timeframe,
)
from finassist.core.config import Settings
from finassist.prompts import MARKET_AGENT_SYSTEM_PROMPT
from finassist.workflow.state import FinanceAssistantState

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = frozenset({"get_quote", "search_symbol", "get_time_series"})
UNAVAILABLE = "I couldn't retrieve a market quote. Please try again later."


@asynccontextmanager
async def twelve_data_session(settings: Settings) -> AsyncIterator[ClientSession]:
    """Keep the transport and session alive for the entire tool conversation."""
    assert settings.twelve_data_api_key is not None
    parameters = StdioServerParameters(
        command=settings.twelve_data_mcp_command,
        args=settings.twelve_data_mcp_args,
        env={"TWELVE_DATA_API_KEY": settings.twelve_data_api_key.get_secret_value()},
    )
    # Provider diagnostics can contain request URLs; do not expose them to API users.
    with open(os.devnull, "w") as diagnostics:  # noqa: ASYNC230
        async with stdio_client(parameters, errlog=diagnostics) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MarketAnalysisAgent:
    def __init__(
        self,
        *,
        llm: Any,
        settings: Settings,
        sessions: MarketSessionManager | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.sessions = sessions

    async def ainvoke(self, state: FinanceAssistantState) -> FinanceAssistantState:
        if not self.settings.market_quotes_enabled:
            return {"answer": "Market quotes are disabled.", "status": "answered"}
        if self.settings.twelve_data_api_key is None:
            return {
                "answer": "Market quotes are unavailable. Set TWELVE_DATA_API_KEY.",
                "status": "error",
            }
        try:
            async with asyncio.timeout(self.settings.market_agent_timeout_seconds):
                connection = (
                    self.sessions.borrow() if self.sessions else twelve_data_session(self.settings)
                )
                async with connection as session:
                    async with market_request_budget(
                        self.settings.market_provider_max_calls_per_request
                    ):
                        return await self._run(state, session)
        except TimeoutError:
            return {
                "answer": "Market data retrieval exceeded the agent's time limit. "
                "Please retry or request a shorter date range.",
                "status": "error",
            }
        except Exception as error:
            logger.warning("market_agent_failed category=%s", type(error).__name__)
            # Never return exception text: upstream errors can contain credentials.
            answer = RATE_LIMIT_RESPONSE if is_rate_limit(error) else UNAVAILABLE
            return {"answer": answer, "status": "error"}

    async def _run(
        self, state: FinanceAssistantState, session: ClientSession
    ) -> FinanceAssistantState:
        nasdaq_only = state.get("market_scope") == "us_nasdaq_usd" or bool(
            re.search(
                r"(?:Market scope|Scope):\s*United States,?\s+NASDAQ-listed equities only",
                state["query"],
                re.IGNORECASE,
            )
        )
        catalog = await session.list_tools()
        sized_tools = {
            tool.name
            for tool in catalog.tools
            if "outputsize" in tool.inputSchema.get("properties", {})
        }
        descriptors: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": deepcopy(tool.inputSchema),
                },
            }
            for tool in catalog.tools
            if tool.name in ALLOWED_TOOLS
        ]
        for descriptor in descriptors:
            function = descriptor["function"]
            if function["name"] == "get_time_series":
                function["description"] += (
                    " Timeframe is resolved automatically from the original request; "
                    "do not infer dates, interval or record counts."
                )
            if function["name"] in {"get_quote", "get_time_series"}:
                function["parameters"].setdefault("properties", {})["finish_response"] = {
                    "type": "boolean",
                    "description": (
                        "Set true on the LAST data call only when all requested assets/data "
                        "will be complete and a standard quote/chart summary is enough. "
                        "Python then renders the response without another model turn. "
                        "Standard OHLC/volume chart data is routine and stays in the structured "
                        "response. Use false only for pending tools/clarification, analysis, "
                        "explanations, custom prose/tables, growth or extrema."
                    ),
                }
                required = function["parameters"].setdefault("required", [])
                if "finish_response" not in required:
                    required.append("finish_response")
        descriptors.append(
            {
                "type": "function",
                "function": {
                    "name": "calculate_price_growth",
                    "description": (
                        "Retrieve prices and calculate single/multi-asset growth. "
                        "Timeframe is resolved automatically. Do not supply prices or dates."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                    },
                                ]
                            }
                        },
                        "required": ["symbol"],
                    },
                },
            }
        )
        descriptors.append(
            {
                "type": "function",
                "function": {
                    "name": "calculate_price_extrema",
                    "description": "Compute a historical minimum/maximum from provider history. "
                    "Resolve the symbol; timeframe is resolved automatically. "
                    "Never supply prices.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "operation": {"type": "string", "enum": ["min", "max"]},
                            "field": {"type": "string", "enum": ["open", "high", "low", "close"]},
                        },
                        "required": ["symbol", "operation"],
                    },
                },
            }
        )
        available = {item["function"]["name"] for item in descriptors}
        if "get_quote" not in available:
            return {"answer": UNAVAILABLE, "status": "error"}
        model = self.llm.bind_tools(descriptors)
        messages = [SystemMessage(MARKET_AGENT_SYSTEM_PROMPT), HumanMessage(state["query"])]
        timeframe: dict[str, object] | None = None
        retrieved_data = False
        observations: list[dict[str, Any]] = []
        for _ in range(5):
            response = await model.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                answer = response.content
                if not isinstance(answer, str) or not answer.strip():
                    return {"answer": UNAVAILABLE, "status": "error"}
                if not retrieved_data:
                    return {"answer": answer, "clarification_question": answer, "status": "clarify"}
                return cast(FinanceAssistantState, market_response(answer, observations))
            calls = response.tool_calls
            if timeframe is None and any(
                c["name"]
                in {"get_time_series", "calculate_price_growth", "calculate_price_extrema"}
                for c in calls
            ):
                timeframe = await resolve_market_timeframe.ainvoke({"request": state["query"]})
                if timeframe["status"] == "clarify":
                    question = str(timeframe["clarification_question"])
                    return {
                        "answer": question,
                        "clarification_question": question,
                        "status": "clarify",
                    }
            last_data_call = next(
                (
                    call
                    for call in reversed(calls)
                    if call["name"] in {"get_quote", "get_time_series"}
                ),
                None,
            )
            finish_snapshot = False
            batch_valid = True
            for call in calls:
                result = None
                if call["name"] == "calculate_price_extrema":
                    if timeframe is None or "start_date" not in cast(
                        dict[str, Any], timeframe["parameters"]
                    ):
                        payload = '{"error": "Resolve a date-bounded historical window first."}'
                    elif "get_time_series" not in available:
                        payload = '{"error": "Historical MCP data is unavailable."}'
                    else:
                        return await calculate_price_extrema(
                            session,
                            call["args"]["symbol"],
                            cast(dict[str, Any], timeframe["parameters"]),
                            call["args"]["operation"],
                            call["args"].get("field"),
                        )
                elif call["name"] == "calculate_price_growth":
                    if timeframe is None or "start_date" not in cast(
                        dict[str, Any], timeframe["parameters"]
                    ):
                        payload = '{"error": "Resolve a growth date window first."}'
                    elif "get_time_series" not in available:
                        payload = '{"error": "Historical MCP data is unavailable."}'
                    else:
                        # Preserve every comparison asset even when growth and timeframe
                        # were selected together in the first turn.
                        symbols: list[str] = []
                        growth_calls = [c for c in calls if c["name"] == "calculate_price_growth"]
                        for growth_call in growth_calls:
                            value = growth_call["args"]["symbol"]
                            symbols.extend([value] if isinstance(value, str) else value)
                        return await calculate_price_growth(
                            session,
                            symbols if len(growth_calls) > 1 else call["args"]["symbol"],
                            cast(dict[str, Any], timeframe["parameters"]),
                            str(timeframe["as_of"]),
                        )
                elif call["name"] == "get_time_series" and call["name"] in available:
                    if timeframe is None:
                        payload = '{"error": "Historical timeframe unavailable."}'
                    else:
                        # The LLM selects the asset; Python owns all window parameters.
                        arguments = {
                            key: value
                            for key, value in call["args"].items()
                            if key in {"symbol", "exchange", "country", "mic_code"}
                        }
                        arguments.update(cast(dict[str, object], timeframe["parameters"]))
                        if nasdaq_only:
                            arguments.update(exchange="NASDAQ", country="United States")
                            arguments.pop("mic_code", None)
                        result = await market_cache.call_tool(session, call["name"], arguments)
                        if is_rate_limit(result):
                            return {"answer": RATE_LIMIT_RESPONSE, "status": "error"}
                        retrieved_data = True
                        payload = compact_tool_result(result)
                elif call["name"] not in available:
                    payload = '{"error": "This tool is not available."}'
                else:
                    arguments = {
                        key: value
                        for key, value in call["args"].items()
                        if key != "finish_response"
                    }
                    if nasdaq_only:
                        arguments.update(exchange="NASDAQ", country="United States")
                        arguments.pop("mic_code", None)
                    if call["name"] in sized_tools:
                        arguments["outputsize"] = min(
                            MAX_RECORDS, max(1, int(arguments.get("outputsize") or MAX_RECORDS))
                        )
                    result = await market_cache.call_tool(session, call["name"], arguments)
                    if is_rate_limit(result):
                        return {"answer": RATE_LIMIT_RESPONSE, "status": "error"}
                    retrieved_data = retrieved_data or call["name"] == "get_quote"
                    payload = compact_tool_result(result)
                if call["name"] in {"get_quote", "get_time_series"} and result is not None:
                    market_observation = chart_observation(
                        result,
                        tool=call["name"],
                        symbol=str(call["args"].get("symbol", "")),
                        interval=str(arguments.get("interval", "1day")),
                    )
                    if market_observation:
                        if nasdaq_only and market_observation.get("currency") not in {None, "USD"}:
                            return {
                                "answer": "The Market page requires USD observations; "
                                "data in another currency cannot be substituted.",
                                "status": "error",
                            }
                        if nasdaq_only and market_observation.get("exchange") not in {
                            None,
                            "NASDAQ",
                            "XNAS",
                        }:
                            return {
                                "answer": (
                                    "No matching NASDAQ listing was returned for this request. "
                                    "The Market page supports NASDAQ-listed equities only; "
                                    "data from another exchange cannot be substituted."
                                ),
                                "status": "error",
                            }
                        observations.append(market_observation)
                        if call["args"].get("finish_response") is True and call is last_data_call:
                            finish_snapshot = True
                    else:
                        batch_valid = False
                elif result is None or getattr(result, "isError", False):
                    # A pending/failed operation must not be hidden by a successful quote.
                    batch_valid = False
                messages.append(ToolMessage(content=payload, tool_call_id=call["id"]))
            if finish_snapshot and batch_valid:
                return cast(
                    FinanceAssistantState,
                    market_response(format_snapshot(observations), observations),
                )
        return {"answer": UNAVAILABLE, "status": "error"}
