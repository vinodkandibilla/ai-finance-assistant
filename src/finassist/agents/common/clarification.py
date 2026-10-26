"""Shared checkpointed clarification support for domain agents."""

from collections.abc import Awaitable, Callable

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from finassist.workflow.state import FinanceAssistantState


def clarification_reply(question: str) -> str:
    """Pause execution using the clarification payload understood by the query API."""
    return str(interrupt({"kind": "clarification", "question": question}))


def build_clarification_graph(
    handler: Callable[[FinanceAssistantState], Awaitable[FinanceAssistantState]],
) -> CompiledStateGraph[FinanceAssistantState, None, FinanceAssistantState, FinanceAssistantState]:
    """Persist domain results before asking, then resume only that domain."""

    def clarify(state: FinanceAssistantState) -> FinanceAssistantState:
        reply = clarification_reply(state.get("clarification_question") or state["answer"])
        return {
            "query": state["query"] + "\nAdditional details: " + reply,
            "answer": "",
            "clarification_question": "",
            "status": "resuming",
        }

    async def run(state: FinanceAssistantState) -> FinanceAssistantState:
        return await handler(state)

    def select_clarification_or_end(state: FinanceAssistantState) -> str:
        return "clarify" if state.get("status") == "clarify" else END

    graph = StateGraph(FinanceAssistantState)
    graph.add_node("run", RunnableLambda(run, name="Execute specialist request"))
    graph.add_node("clarify", RunnableLambda(clarify, name="Ask for missing details"))
    graph.add_edge(START, "run")
    graph.add_conditional_edges(
        "run", RunnableLambda(select_clarification_or_end, name="Clarify or finish")
    )
    graph.add_edge("clarify", "run")
    return graph.compile()
