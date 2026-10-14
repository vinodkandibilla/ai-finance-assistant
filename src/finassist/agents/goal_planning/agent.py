"""Model-directed goal planning with deterministic calculator tools."""

import asyncio
import json
import math

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import Field

from finassist.agents.common.clarification import clarification_reply
from finassist.agents.common.schemas import AgentResult, AgentState
from finassist.agents.goal_planning.tools import (
    calculate_future_value,
    calculate_required_contribution,
    calculate_time_to_goal,
    check_goal_feasibility,
)
from finassist.prompts import (
    GOAL_PLANNING_AGENT_PROMPT as AGENT_PROMPT,
    GOAL_PLANNING_TOOL_GUIDANCE as TOOL_GUIDANCE,
)


class GoalPlanningResult(AgentResult):
    sufficient: bool = Field(description="Tool results cover all requested parts of the task.")
    aligned: bool = Field(description="Tool operation and inputs match the user's current goal.")
    next_action: str = Field(
        default="",
        description="Describe needed clarification or tool call; do not recalculate arithmetic.",
    )


CALCULATORS = (
    calculate_required_contribution,
    calculate_future_value,
    calculate_time_to_goal,
    check_goal_feasibility,
)
CALCULATOR_NAMES = {calculator.name for calculator in CALCULATORS}
MAX_ITERATIONS = 3
ITERATION_LIMIT_RESPONSE = (
    "I could not complete your goal-planning request within three iterations with the "
    "available information or capabilities. Please submit a new request with your target, "
    "current savings, monthly contribution or deadline, and expected annual return "
    "if you want to include growth."
)


class _IterationLimitReached(Exception):
    def __init__(self, messages):
        self.messages = messages


def _iteration_count(messages) -> int:
    # Model turns are persisted in the child graph, including across clarification.
    return sum(isinstance(message, AIMessage) for message in messages)


def _check_iteration_budget(state):
    if _iteration_count(state["messages"]) >= MAX_ITERATIONS:
        raise _IterationLimitReached(state["messages"])
    return {"llm_input_messages": state["messages"]}


@tool
def ask_goal_clarification(question: str) -> str:
    """Ask for essential missing goal details and wait for the user's reply."""
    return clarification_reply(question)


class GoalPlanningAgent:
    def __init__(
        self,
        *,
        llm,
        checkpointer=None,
        timeout_seconds: float = 60,
        recursion_limit: int = 30,
    ):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if recursion_limit < 6:
            raise ValueError("recursion_limit must be at least 6")
        self.timeout_seconds = timeout_seconds
        self.recursion_limit = recursion_limit
        self.agent = create_react_agent(
            model=llm,
            tools=ToolNode([*CALCULATORS, ask_goal_clarification], handle_tool_errors=True),
            prompt=AGENT_PROMPT + "\n" + TOOL_GUIDANCE,
            name="goal_planning_agent",
            pre_model_hook=_check_iteration_budget,
        )
        self._formatter = llm.with_structured_output(GoalPlanningResult)
        workflow = StateGraph(AgentState)
        workflow.add_node("goal_planning_agent", self._run)
        workflow.add_edge(START, "goal_planning_agent")
        workflow.add_edge("goal_planning_agent", END)
        self.graph = workflow.compile(checkpointer=checkpointer)

    async def ainvoke(self, state: dict, config: dict | None = None) -> dict:
        return await self.graph.ainvoke(state, config=config)

    async def _run(self, state: AgentState, config) -> dict:
        calculations = []
        try:
            async with asyncio.timeout(self.timeout_seconds):
                messages = [HumanMessage(content=state["query"])]
                # A failed coverage check returns to the same tool loop with its history.
                for _attempt in range(MAX_ITERATIONS):
                    try:
                        output = await self.agent.ainvoke(
                            {"messages": messages},
                            config={**config, "recursion_limit": self.recursion_limit},
                        )
                    except _IterationLimitReached as limit:
                        # Evaluate the third turn's results, without another decision/tool turn.
                        output = {"messages": limit.messages}
                    assessment = GoalPlanningResult.model_validate(
                        await self._formatter.ainvoke(
                            [
                                SystemMessage(content=AGENT_PROMPT + "\n" + TOOL_GUIDANCE),
                                *output["messages"],
                                HumanMessage(
                                    content="Check coverage and alignment with the original goal "
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
                            content="The goal response is incomplete or misaligned. "
                            + assessment.next_action
                            + " Continue the same request: ask for missing "
                            "details through ask_goal_clarification or call the needed calculator. "
                            "Do not recalculate arithmetic yourself."
                        ),
                    ]
                result = AgentResult.model_validate(assessment.model_dump())
                if result.status == "answered" and not (
                    assessment.sufficient and assessment.aligned
                ):
                    result = AgentResult(
                        status="insufficient",
                        answer="The available projection does not yet fully address your goal.",
                    )
                # Keep the last attempt for each tool, so repaired failures are superseded.
                latest = {}
                for message in output["messages"]:
                    if isinstance(message, ToolMessage) and message.name in CALCULATOR_NAMES:
                        try:
                            payload = json.loads(message.content)
                        except (TypeError, ValueError):
                            payload = {"status": "error", "code": "TOOL_ERROR"}
                        latest[message.name] = payload
                calculations = list(latest.values())
                if result.status == "answered" and (
                    not calculations or any(c.get("status") != "success" for c in calculations)
                ):
                    result = AgentResult(
                        status="insufficient",
                        answer="I could not complete the projection with valid calculator results.",
                    )
                if result.status == "clarification":
                    result = AgentResult(
                        status="insufficient", answer="Essential goal details are still missing."
                    )
                if result.status == "answered" and not result.answer.strip():
                    result = AgentResult(status="error", answer="Goal answer generation failed.")
                if _iteration_count(output["messages"]) >= MAX_ITERATIONS and result.status not in {
                    "answered",
                    "out_of_scope",
                    "error",
                }:
                    result = AgentResult(
                        status="insufficient",
                        answer=ITERATION_LIMIT_RESPONSE,
                        errors=["goal_iteration_limit"],
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
                answer="Goal planning reached its execution limit. Please try again.",
                errors=[type(error).__name__],
            )
        except Exception as error:
            calculations = []
            result = AgentResult(
                status="error",
                answer="Goal planning is currently unavailable.",
                errors=[type(error).__name__],
            )
        supplemental = {}
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
            "intent": "goals",
        }
