"""Model-directed portfolio analysis with deterministic calculator tools."""

import asyncio
import json
import math
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command
from pydantic import Field

from finassist.agents.common.clarification import clarification_reply
from finassist.agents.common.schemas import AgentResult, AgentState
from finassist.agents.portfolio_analysis.authorization import UserContext, denied
from finassist.agents.portfolio_analysis.tools import PortfolioTools
from finassist.data.portfolio_store import PortfolioStore
from finassist.prompts import PORTFOLIO_ANALYSIS_AGENT_PROMPT as AGENT_PROMPT

TOOL_GUIDANCE = (
    "Use deterministic tool results; never recalculate metrics. "
    "For routine portfolio listings, screens or dashboard retrieval, explicitly set "
    "finish_response=true on screen_manager_portfolios or compare_manager_portfolios "
    "when that single tool covers the entire request. Use false when interpretation, "
    "profile alignment or additional tools are needed."
)


class PortfolioState(AgentState, total=False):
    customer_id: str
    manager_id: str
    user_role: str


class PortfolioAnalysisResult(AgentResult):
    profile_alignment_customer_ids: list[str] = Field(
        default_factory=list,
        description="Customer IDs needing profile-alignment interpretation in the latest request.",
    )
    sufficient: bool = Field(description="Tool results cover all requested parts of the task.")
    aligned: bool = Field(description="Tool operation and inputs match the user's current goal.")
    next_action: str = Field(
        default="",
        description="Describe needed clarification or tool call; do not recalculate arithmetic.",
    )


MAX_ITERATIONS = 6
ITERATION_LIMIT_RESPONSE = (
    "I could not complete the portfolio analysis within the execution limit. "
    "Please provide the customer or manager identifier and the analysis you need."
)


class _IterationLimitReached(Exception):
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


class _CompletedScreen(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


def _iteration_count(messages: list[Any]) -> int:
    # Model turns are persisted in the child graph, including across clarification.
    return sum(isinstance(message, AIMessage) for message in messages)


def _check_iteration_budget(state: dict[str, Any]) -> dict[str, Any]:
    messages = state["messages"]
    calls = [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    if len(calls) == 1 and calls[0]["name"] in {
        "screen_manager_portfolios",
        "compare_manager_portfolios",
    }:
        last = messages[-1]
        if (
            calls[0]["args"].get("finish_response") is True
            and isinstance(last, ToolMessage)
            and last.tool_call_id == calls[0]["id"]
        ):
            try:
                payload = json.loads(str(last.content))
            except (TypeError, ValueError):
                payload = None
            if (
                isinstance(payload, dict)
                and payload.get("status") == "success"
                and isinstance(payload.get("portfolios"), list)
            ):
                raise _CompletedScreen(payload)
    if _iteration_count(state["messages"]) >= MAX_ITERATIONS:
        raise _IterationLimitReached(state["messages"])
    return {"llm_input_messages": state["messages"]}


@tool
def ask_portfolio_clarification(question: str) -> str:
    """Ask for essential missing portfolio identifiers and wait for the user's reply."""
    return clarification_reply(question)


class PortfolioAnalysisAgent:
    def __init__(
        self,
        *,
        llm: Any,
        store: PortfolioStore,
        user_context: UserContext | None = None,
        portfolio_tools: PortfolioTools | None = None,
        checkpointer: Any = None,
        timeout_seconds: float = 60,
        recursion_limit: int = 30,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if recursion_limit < 6:
            raise ValueError("recursion_limit must be at least 6")
        self.timeout_seconds = timeout_seconds
        self.recursion_limit = recursion_limit
        self.tools = portfolio_tools or PortfolioTools(store, user_context=user_context)
        identity_prompt = ""
        if self.tools.user_context:
            identity_prompt = "\nApplication identity: " + self.tools.user_context.model_dump_json()
        tools = self.tools.as_tools()
        self._tool_names = {t.name for t in tools}
        self.agent = create_react_agent(
            model=llm,
            tools=ToolNode([*tools, ask_portfolio_clarification], handle_tool_errors=True),
            prompt=AGENT_PROMPT + "\n" + TOOL_GUIDANCE + identity_prompt,
            name="portfolio_analysis_agent",
            pre_model_hook=_check_iteration_budget,
        )
        self._formatter = llm.with_structured_output(PortfolioAnalysisResult).with_config(
            run_name="Assess portfolio answer"
        )
        workflow = StateGraph(PortfolioState)
        workflow.add_node("portfolio_analysis_agent", self._run)
        workflow.add_edge(START, "portfolio_analysis_agent")
        workflow.add_edge("portfolio_analysis_agent", END)
        self.graph = workflow.compile(checkpointer=checkpointer)

    async def ainvoke(
        self, state: PortfolioState | Command[Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return await self.graph.ainvoke(state, config=config)

    async def _run(self, state: PortfolioState, config: RunnableConfig) -> dict[str, Any]:
        calculations = []
        try:
            async with asyncio.timeout(self.timeout_seconds):
                context = {
                    k: cast(dict[str, Any], state)[k]
                    for k in ("customer_id", "manager_id", "user_role")
                    if state.get(k)
                }
                trusted = config.get("configurable", {}).get("portfolio_user_context")
                if trusted is None and self.tools.user_context is None:
                    return {
                        "intent": "portfolio",
                        "status": "error",
                        "answer": denied()["message"],
                        "errors": ["ACCESS_DENIED"],
                    }
                messages = [
                    HumanMessage(
                        content=state["query"]
                        + "\nRequested scope (not identity): "
                        + json.dumps(context)
                        + "\nApplication identity: "
                        + json.dumps(
                            trusted
                            or (
                                self.tools.user_context.model_dump()
                                if self.tools.user_context
                                else {}
                            )
                        )
                    )
                ]
                # A failed coverage check returns to the same tool loop with its history.
                for _attempt in range(MAX_ITERATIONS):
                    try:
                        output = await self.agent.ainvoke(
                            {"messages": messages},
                            config=cast(
                                RunnableConfig, {**config, "recursion_limit": self.recursion_limit}
                            ),
                        )
                    except _CompletedScreen as completed:
                        payload = completed.payload
                        rows = payload["portfolios"]
                        names = ", ".join(str(row["customer_id"]) for row in rows) or "None"
                        excluded_count = len(payload.get("excluded_portfolios", []))
                        result = AgentResult(
                            status="answered",
                            answer=(
                                f"Screen: {payload.get('screen', 'all')}; "
                                f"ranked by {payload['rank_by']}. "
                                f"Matching portfolios ({len(rows)}): {names}. "
                                f"Excluded portfolios: {excluded_count}. "
                                f"Source: {payload['data_source']}. "
                                f"Price basis: {payload['price_basis']}."
                            ),
                        )
                        return {
                            **result.model_dump(),
                            "calculation": payload,
                            "final_result": result.model_dump(),
                            "intent": "portfolio",
                        }
                    except _IterationLimitReached as limit:
                        # Evaluate the sixth turn's results, without another decision/tool turn.
                        output = {"messages": limit.messages}
                    assessment = PortfolioAnalysisResult.model_validate(
                        await self._formatter.ainvoke(
                            [
                                SystemMessage(content=AGENT_PROMPT + "\n" + TOOL_GUIDANCE),
                                *output["messages"],
                                HumanMessage(
                                    content="Check coverage and alignment with the request "
                                    "and user replies. Do not recalculate tool math. "
                                    "If incomplete, specify the next clarification or tool action."
                                ),
                            ]
                        )
                    )
                    if (
                        assessment.status in {"out_of_scope", "error"}
                        or (
                            assessment.status == "answered"
                            and assessment.sufficient
                            and assessment.aligned
                        )
                        or (
                            assessment.status == "insufficient"
                            and not assessment.next_action.strip()
                        )
                    ):
                        break
                    if _iteration_count(output["messages"]) >= MAX_ITERATIONS:
                        break
                    messages = [
                        *output["messages"],
                        HumanMessage(
                            content="The portfolio response is incomplete or misaligned. "
                            + assessment.next_action
                            + " Continue the same request: ask for missing "
                            "details through ask_portfolio_clarification or call a portfolio tool. "
                            "Do not recalculate arithmetic yourself."
                        ),
                    ]
                result = AgentResult.model_validate(assessment.model_dump())
                if result.status == "answered" and not (
                    assessment.sufficient and assessment.aligned
                ):
                    result = AgentResult(
                        status="insufficient",
                        answer="The analysis does not yet fully address your portfolio request.",
                    )
                # Keep the last attempt for each tool, so repaired failures are superseded.
                latest = {}
                arguments = {
                    call["id"]: json.dumps(call["args"], sort_keys=True)
                    for message in output["messages"]
                    if isinstance(message, AIMessage)
                    for call in message.tool_calls
                }
                for message in output["messages"]:
                    if isinstance(message, ToolMessage) and message.name in self._tool_names:
                        try:
                            payload = json.loads(str(message.content))
                        except (TypeError, ValueError):
                            payload = {"status": "error", "code": "TOOL_ERROR"}
                        latest[
                            (
                                message.name,
                                arguments.get(message.tool_call_id, message.tool_call_id),
                            )
                        ] = payload
                calculations = list(latest.values())
                if result.status == "answered" and (
                    not any(
                        "holdings" in c or "profile" in c or "portfolios" in c for c in calculations
                    )
                    or any(c.get("status") != "success" for c in calculations)
                ):
                    result = AgentResult(
                        status="insufficient",
                        answer="Valid portfolio tool results are needed to complete this analysis.",
                    )
                if result.status == "answered" and assessment.profile_alignment_customer_ids:
                    portfolio_ids = {c.get("customer_id") for c in calculations if "holdings" in c}
                    profile_ids = {
                        c["profile"]["customer_id"] for c in calculations if "profile" in c
                    }
                    if (
                        not set(assessment.profile_alignment_customer_ids)
                        <= portfolio_ids & profile_ids
                    ):
                        result = AgentResult(
                            status="insufficient",
                            answer=(
                                "Alignment requires portfolio metrics and the stored profile "
                                "for the same customer."
                            ),
                        )
                if any(c.get("code") == "ACCESS_DENIED" for c in calculations):
                    result = AgentResult(
                        status="error", answer=denied()["message"], errors=["ACCESS_DENIED"]
                    )
                    calculations = []
                if result.status == "clarification":
                    result = AgentResult(
                        status="insufficient",
                        answer="Essential portfolio identifiers are still missing.",
                    )
                if result.status == "answered" and not result.answer.strip():
                    result = AgentResult(
                        status="error", answer="Portfolio answer generation failed."
                    )
                if _iteration_count(output["messages"]) >= MAX_ITERATIONS and result.status not in {
                    "answered",
                    "out_of_scope",
                    "error",
                }:
                    result = AgentResult(
                        status="insufficient",
                        answer=ITERATION_LIMIT_RESPONSE,
                        errors=["portfolio_iteration_limit"],
                    )
                if result.status == "out_of_scope":
                    calculations = []
                result.citations = []
                result.clarification_question = None
        except GraphInterrupt:
            raise
        except (TimeoutError, GraphRecursionError) as error:
            calculations = []
            result = AgentResult(
                status="error",
                answer="Portfolio analysis reached its execution limit. Please try again.",
                errors=[type(error).__name__],
            )
        except Exception as error:
            calculations = []
            result = AgentResult(
                status="error",
                answer="Portfolio analysis is currently unavailable.",
                errors=[type(error).__name__],
            )
        supplemental: dict[str, Any] = {}
        if calculations:
            supplemental = (
                {"calculation": calculations[0]}
                if len(calculations) == 1
                else {"calculations": calculations}
            )
        return {
            **result.model_dump(),
            **supplemental,
            "final_result": result.model_dump(),
            "intent": "portfolio",
        }
