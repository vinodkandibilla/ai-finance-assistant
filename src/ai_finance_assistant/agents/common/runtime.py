"""Shared tool execution and API adapter; the model chooses the action sequence."""

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import interrupt

from ai_finance_assistant.agents.common.schemas import AgentResult, AgentState, EvidenceResult
from ai_finance_assistant.rag.models import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentPolicy:
    intent: str
    label: str
    name: str
    prompt: str
    search_tool_name: str
    clarification_tool_name: str
    no_evidence_answer: str
    citations: Callable[[list[RetrievedDocument]], list[str]]
    search_arguments: Callable[[str, str, list[str], list[str]], dict]
    clarification_response: Callable[[str, str], str | None] | None = None
    process_tool_results: Callable | None = None
    additional_reply_tools: tuple[str, ...] = ()


def user_replies(messages, tool_name, additional_tools=()):
    """Return genuine user replies, excluding tool policy guidance."""
    return [
        str(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name in {tool_name, *additional_tools}
        and (message.artifact or {}).get("kind") == "user_reply"
    ]


class AutonomousEducationAgent:
    def __init__(
        self,
        *,
        llm,
        retrieval,
        settings,
        policy: AgentPolicy,
        result_schema: type[AgentResult],
        checkpointer=None,
        additional_tools=(),
    ):
        self._settings = settings
        self._policy = policy
        self._result_schema = result_schema
        self._checkpointer = checkpointer

        @tool(policy.search_tool_name, response_format="content_and_artifact")
        async def search(
            query: str,
            state: Annotated[dict, InjectedState],
            dense_queries: list[str] | None = None,
            sparse_keywords: list[str] | None = None,
        ) -> tuple[dict, dict]:
            """Search domain evidence and rerank passages; returns valid citations and errors."""
            calls = sum(
                sum(
                    call["name"] == policy.search_tool_name
                    for call in getattr(message, "tool_calls", [])
                )
                for message in state["messages"]
            )
            if calls > settings.agent_max_searches:
                result = EvidenceResult(errors=[f"{policy.label} search budget exhausted."])
            else:
                original = str(state["messages"][0].content)
                for reply in user_replies(
                    state["messages"], policy.clarification_tool_name, policy.additional_reply_tools
                ):
                    original += "\nAdditional details: " + reply
                try:
                    result = EvidenceResult.model_validate(
                        await retrieval.ainvoke(
                            policy.search_arguments(
                                query,
                                original,
                                dense_queries or [query],
                                sparse_keywords if sparse_keywords is not None else [query],
                            )
                        )
                    )
                except Exception as error:
                    result = EvidenceResult(
                        errors=[f"{policy.label} search failed: {type(error).__name__}"]
                    )
            payload = result.model_dump()
            return payload, payload

        @tool(policy.clarification_tool_name, response_format="content_and_artifact")
        def clarify(question: str, state: Annotated[dict, InjectedState]) -> tuple[str, dict]:
            """Ask for essential missing details and wait for the user's reply."""
            if re.search(
                r"\b(?:submit|start|ask|send|provide)\b.*\bnew question\b", question, re.I
            ):
                return (
                    "A request to submit a new question is not a missing-detail clarification. "
                    "If the user's changed task is outside your scope, finish with out_of_scope. "
                    "Do not pause the task again.",
                    {"kind": "policy_guidance"},
                )
            original = str(state["messages"][0].content)
            for reply in user_replies(
                state["messages"], policy.clarification_tool_name, policy.additional_reply_tools
            ):
                original += "\nAdditional details: " + reply
            guidance = (
                policy.clarification_response(question, original)
                if policy.clarification_response
                else None
            )
            if guidance is not None:
                return guidance, {"kind": "policy_guidance"}
            reply = str(interrupt({"kind": "clarification", "question": question}))
            return reply, {"kind": "user_reply"}

        def agent_prompt(state):
            messages = [
                SystemMessage(
                    content=policy.prompt + f"\nSearch budget: {settings.agent_max_searches} calls."
                ),
                *state["messages"],
            ]
            replies = user_replies(
                state["messages"], policy.clarification_tool_name, policy.additional_reply_tools
            )
            if replies:
                messages.append(
                    HumanMessage(
                        content="My latest follow-up is: "
                        + replies[-1]
                        + "\nIf this replaces my original task, handle this changed task. "
                        "If it supplies missing details, continue the original task."
                    )
                )
            return messages

        self.retrieval_tool = search
        self.agent = create_react_agent(
            model=llm,
            tools=[search, clarify, *additional_tools],
            prompt=agent_prompt,
            name=policy.name,
        )
        self._formatter = llm.with_structured_output(result_schema)
        self.graph = self._build_graph()

    async def ainvoke(self, state: dict, config: dict) -> dict:
        return await self.graph.ainvoke(state, config=config)

    def _build_graph(self):
        policy = self._policy
        schema = self._result_schema

        async def run_agent(state: AgentState, config):
            query = str(state.get("search_query") or state.get("user_query") or state["query"])
            supplemental = {}
            started = time.monotonic()
            deadline = asyncio.timeout(self._settings.agent_timeout_seconds)
            try:
                async with deadline:
                    output = await self.agent.ainvoke(
                        {"messages": [HumanMessage(content=query)]},
                        config={**config, "recursion_limit": self._settings.agent_recursion_limit},
                    )
                    replies = user_replies(
                        output["messages"],
                        policy.clarification_tool_name,
                        policy.additional_reply_tools,
                    )
                    final = await self._formatter.ainvoke(
                        [
                            SystemMessage(
                                content=policy.prompt
                                + "\nFormat the completed task. A latest reply replacing the task "
                                "outside your scope requires out_of_scope, not insufficient. "
                                "Clarification must use the interrupt tool."
                            ),
                            *output["messages"],
                            HumanMessage(
                                content="Original question: "
                                + query
                                + "\nActual clarification replies: "
                                + repr(replies)
                                + "\nHonor the latest actual user reply if it changes the task. "
                                "Return the outcome for that task, not the abandoned original task."
                            ),
                        ]
                    )
                    result = schema.model_validate(final)
                    evidence, errors = [], []
                    for message in output["messages"]:
                        if (
                            isinstance(message, ToolMessage)
                            and message.name == policy.search_tool_name
                        ):
                            artifact = EvidenceResult.model_validate(message.artifact or {})
                            evidence.extend(artifact.candidates)
                            errors.extend(artifact.errors)
                    supplemental = {}
                    trusted_calculation = False
                    if policy.process_tool_results:
                        processed = policy.process_tool_results(
                            result, output["messages"], evidence
                        )
                        if inspect.isawaitable(processed):
                            processed = await processed
                        result, supplemental, trusted_calculation = processed
                    available = policy.citations(evidence)
                    if result.status == "answered" and not evidence and not trusted_calculation:
                        result = schema(
                            status="error" if errors else "insufficient",
                            answer=f"{policy.label} education retrieval is currently unavailable."
                            if errors
                            else policy.no_evidence_answer,
                            errors=errors,
                        )
                    elif result.status == "answered":
                        result.citations = [
                            c for c in result.citations if c in available
                        ] or available
                        if not result.answer.strip():
                            result = schema(
                                status="error", answer=f"{policy.label} answer generation failed."
                            )
                    else:
                        result.citations = []
                    if result.status == "out_of_scope":
                        result.answer = (
                            f"This request is outside my {policy.label.lower()}-education scope. "
                            "Please submit it as a new question."
                        )
                        result.clarification_question = None
            except GraphInterrupt:
                raise
            except TimeoutError:
                if not deadline.expired():
                    result = schema(
                        status="error",
                        answer=f"{policy.label} agent is currently unavailable.",
                        errors=["TimeoutError"],
                    )
                else:
                    elapsed_ms = (time.monotonic() - started) * 1000
                    timeout_ms = self._settings.agent_timeout_seconds * 1000
                    logger.warning(
                        "agent_execution_failed agent=%s category=timeout elapsed_ms=%.3f "
                        "timeout_ms=%.3f",
                        policy.name,
                        elapsed_ms,
                        timeout_ms,
                        extra={
                            "agent_name": policy.name,
                            "failure_category": "timeout",
                            "elapsed_ms": elapsed_ms,
                            "timeout_ms": timeout_ms,
                        },
                    )
                    supplemental = {}
                    result = schema(
                        status="error",
                        answer=f"{policy.label} agent execution timed out. Please try again.",
                        errors=["agent_timeout"],
                    )
            except GraphRecursionError:
                result = schema(
                    status="error", answer=f"{policy.label} agent reached its execution limit."
                )
            except Exception as error:
                result = schema(
                    status="error",
                    answer=f"{policy.label} agent is currently unavailable.",
                    errors=[type(error).__name__],
                )
            return {
                **result.model_dump(),
                **supplemental,
                "final_result": result.model_dump(),
                "user_query": query,
                "search_query": query,
                "intent": policy.intent,
            }

        workflow = StateGraph(AgentState)
        workflow.add_node(policy.intent + "_agent", run_agent)
        workflow.add_edge(START, policy.intent + "_agent")
        workflow.add_edge(policy.intent + "_agent", END)
        return workflow.compile(checkpointer=self._checkpointer)
