"""Safe request-side LangSmith instrumentation, not MCP-subprocess HTTP tracing."""

import re
from typing import Any

from langsmith import get_current_run_tree, traceable
from mcp import ClientSession
from mcp.types import CallToolResult

from finassist.agents.market_analysis.chart_data import provider_payload
from finassist.agents.market_analysis.retry import MarketMCPError, transient_exception
from finassist.core.config import get_settings

PUBLIC_SELECTORS = {
    "symbol",
    "interval",
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
    "mic_code",
}


def _redact(value: str) -> str:
    settings = get_settings()
    for key in (settings.openai_api_key, settings.twelve_data_api_key, settings.tavily_api_key):
        if key:
            value = value.replace(key.get_secret_value(), "[redacted]")
    value = re.sub(r"https?://\S+", "[url removed]", value)
    value = re.sub(
        r"(?i)(api[_ -]?key|token|authorization|bearer|secret)\s*[:= ]\s*\S+",
        r"\1=[redacted]",
        value,
    )
    return value[:400]


def tool_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": inputs.get("name"),
        "arguments": {
            key: _redact(value) if isinstance(value, str) else value
            for key, value in inputs.get("arguments", {}).items()
            if key in PUBLIC_SELECTORS and isinstance(value, (str, int, float, bool))
        },
    }


def tool_output(result: CallToolResult) -> dict[str, Any]:
    data = provider_payload(result)
    error = bool(result.isError) or (isinstance(data, dict) and data.get("status") == "error")
    summary: dict[str, Any] = {"isError": error}
    if error:
        if isinstance(data, dict):
            for key in ("code", "status", "message"):
                if key in data and isinstance(data[key], (str, int)):
                    summary[key] = _redact(str(data[key]))
        elif isinstance(data, str):
            summary["message"] = _redact(data)
        else:
            summary["message"] = "Provider tool reported an error"
    else:
        rows = data.get("values", [data]) if isinstance(data, dict) else data
        summary["record_count"] = len(rows) if isinstance(rows, list) else 0
    return summary


def cache_status(status: str) -> None:
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({"cache_status": status})


@traceable(
    name="Call market MCP tool",
    run_type="tool",
    process_inputs=tool_inputs,
    process_outputs=tool_output,
)
async def invoke_mcp_tool(
    session: ClientSession, name: str, arguments: dict[str, Any]
) -> CallToolResult:
    """One span per actual MCP attempt, with no session, credentials or full data."""
    try:
        return await session.call_tool(name, arguments)
    except TimeoutError:
        raise TimeoutError("Market MCP request timed out") from None
    except Exception as error:
        # Preserve retry eligibility without exposing raw provider messages in traces.
        raise MarketMCPError(
            f"Market MCP transport failed: {type(error).__name__}",
            retryable=transient_exception(error),
        ) from None
