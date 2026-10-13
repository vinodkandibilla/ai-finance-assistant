from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from ai_finance_assistant.main import create_app, lifespan


class CheckpointState(TypedDict, total=False):
    answer: str


async def test_app_checkpoint_supports_async_interrupt_resume_and_restart(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(
        "ai_finance_assistant.main.get_settings",
        lambda: type(
            "Settings", (), {"finance_checkpoint_path": str(checkpoint_path), "log_level": "INFO"}
        )(),
    )
    app = create_app()
    workflow = StateGraph(CheckpointState)

    def clarify(state: CheckpointState) -> CheckpointState:
        return {"answer": interrupt({"question": "Which funds?"})}

    workflow.add_node("clarify", clarify)
    workflow.add_edge(START, "clarify")
    workflow.add_edge("clarify", END)
    config = {"configurable": {"thread_id": "persistent-clarification"}}
    async with lifespan(app):
        graph = workflow.compile(checkpointer=app.state.query_checkpointer)
        result = await graph.ainvoke({}, config=config)
        assert result["__interrupt__"][0].value["question"] == "Which funds?"
        snapshot = await graph.aget_state(config)
        assert snapshot.tasks[0].interrupts
    assert not hasattr(app.state, "query_checkpointer")

    async with lifespan(app):
        graph = workflow.compile(checkpointer=app.state.query_checkpointer)
        result = await graph.ainvoke(Command(resume="ETFs"), config=config)
        assert result["answer"] == "ETFs"
