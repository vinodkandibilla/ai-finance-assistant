"""Quote and historical market agent backed by the official Twelve Data MCP server."""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from finassist.agents.market_analysis.cache import market_cache
from finassist.agents.market_analysis.chart_data import chart_observation
from finassist.agents.market_analysis.growth import (
    RATE_LIMIT_RESPONSE,
    calculate_price_growth,
    is_rate_limit,
)
from finassist.agents.market_analysis.timeframe import (
    MAX_RECORDS,
    resolve_market_timeframe,
)
from finassist.core.config import Settings
from finassist.prompts import MARKET_AGENT_SYSTEM_PROMPT
from finassist.workflow.state import FinanceAssistantState

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
    def __init__(self, *, llm: Any, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

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
                async with twelve_data_session(self.settings) as session:
                    return await self._run(state, session)
        except Exception as error:
            # Never return exception text: upstream errors can contain credentials.
            answer = RATE_LIMIT_RESPONSE if is_rate_limit(error) else UNAVAILABLE
            return {"answer": answer, "status": "error"}

    async def _run(
        self, state: FinanceAssistantState, session: ClientSession
    ) -> FinanceAssistantState:
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
                    "parameters": tool.inputSchema,
                },
            }
            for tool in catalog.tools
            if tool.name in ALLOWED_TOOLS
        ]
        descriptors.append(convert_to_openai_tool(resolve_market_timeframe))
        descriptors.append(
            {
                "type": "function",
                "function": {
                    "name": "calculate_price_growth",
                    "description": (
                        "Retrieve prices and calculate single/multi-asset growth. "
                        "Call resolve_market_timeframe first. Do not supply prices or dates."
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
                for observation in observations:
                    quote = next(
                        (
                            item
                            for item in observations
                            if item["interval"] == "quote"
                            and item["symbol"] == observation["symbol"]
                        ),
                        {},
                    )
                    for field in ("currency", "exchange"):
                        if not observation.get(field) and quote.get(field):
                            observation[field] = quote[field]
                return {
                    "answer": answer,
                    "status": "answered",
                    **({"calculations": observations} if observations else {}),
                }
            growth_calls = [
                call for call in response.tool_calls if call["name"] == "calculate_price_growth"
            ]
            if len(growth_calls) > 1 and timeframe is not None:
                symbols: list[str] = []
                for call in growth_calls:
                    value = call["args"]["symbol"]
                    symbols.extend([value] if isinstance(value, str) else value)
                if "start_date" in cast(dict[str, Any], timeframe["parameters"]):
                    return await calculate_price_growth(
                        session,
                        symbols,
                        cast(dict[str, Any], timeframe["parameters"]),
                        str(timeframe["as_of"]),
                    )
            for call in response.tool_calls:
                result = None
                if call["name"] == "resolve_market_timeframe":
                    timeframe = await resolve_market_timeframe.ainvoke({"request": state["query"]})
                    if timeframe["status"] == "clarify":
                        question = str(timeframe["clarification_question"])
                        return {
                            "answer": question,
                            "clarification_question": question,
                            "status": "clarify",
                        }
                    payload = json.dumps(timeframe)
                elif call["name"] == "calculate_price_growth":
                    if timeframe is None or "start_date" not in cast(
                        dict[str, Any], timeframe["parameters"]
                    ):
                        payload = '{"error": "Resolve a growth date window first."}'
                    elif "get_time_series" not in available:
                        payload = '{"error": "Historical MCP data is unavailable."}'
                    else:
                        return await calculate_price_growth(
                            session,
                            call["args"]["symbol"],
                            cast(dict[str, Any], timeframe["parameters"]),
                            str(timeframe["as_of"]),
                        )
                elif call["name"] == "get_time_series" and call["name"] in available:
                    if timeframe is None:
                        payload = '{"error": "Call resolve_market_timeframe first."}'
                    else:
                        # The LLM selects the asset; Python owns all window parameters.
                        arguments = {
                            key: value
                            for key, value in call["args"].items()
                            if key in {"symbol", "exchange", "country", "mic_code"}
                        }
                        arguments.update(cast(dict[str, object], timeframe["parameters"]))
                        result = await market_cache.call_tool(session, call["name"], arguments)
                        if is_rate_limit(result):
                            return {"answer": RATE_LIMIT_RESPONSE, "status": "error"}
                        retrieved_data = True
                        payload = json.dumps(result.model_dump(mode="json"))
                elif call["name"] not in available:
                    payload = '{"error": "This tool is not available."}'
                else:
                    arguments = dict(call["args"])
                    if call["name"] in sized_tools:
                        arguments["outputsize"] = min(
                            MAX_RECORDS, max(1, int(arguments.get("outputsize") or MAX_RECORDS))
                        )
                    result = await market_cache.call_tool(session, call["name"], arguments)
                    if is_rate_limit(result):
                        return {"answer": RATE_LIMIT_RESPONSE, "status": "error"}
                    retrieved_data = retrieved_data or call["name"] == "get_quote"
                    payload = json.dumps(result.model_dump(mode="json"))
                if call["name"] in {"get_quote", "get_time_series"} and result is not None:
                    observation = chart_observation(
                        result,
                        tool=call["name"],
                        symbol=str(call["args"].get("symbol", "")),
                        interval=str(arguments.get("interval", "1day")),
                    )
                    if observation:
                        observations.append(observation)
                messages.append(ToolMessage(content=payload, tool_call_id=call["id"]))
        return {"answer": UNAVAILABLE, "status": "error"}
