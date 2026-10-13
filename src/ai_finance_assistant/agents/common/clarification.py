"""Shared checkpointed clarification support for domain agents."""

from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from ai_finance_assistant.graph.state import FinanceAssistantState


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

    graph = StateGraph(FinanceAssistantState)
    graph.add_node("run", run)
    graph.add_node("clarify", clarify)
    graph.add_edge(START, "run")
    graph.add_conditional_edges(
        "run", lambda state: "clarify" if state.get("status") == "clarify" else END
    )
    graph.add_edge("clarify", "run")
    return graph.compile()
