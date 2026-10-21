import json
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from finassist.agents.finance_qa.calculator_tools import CLARIFICATION
from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.graph import FinanceQaAgent
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.schemas import FinanceAgentResult, FinanceDocument
from finassist.workflow.builder import build_graph
from test_finance_autonomous_agent import Classifier, Reranker, Retriever, ScriptedModel, call

INPUTS = {"principal": "10000", "annual_rate": "6", "rate_units": "percent", "duration_years": "5"}
Model = ScriptedModel


def graph(model, documents=(), checkpointer=None):
    agent = FinanceQaAgent(
        llm=model,
        retriever=Retriever(list(documents)),
        reranker=Reranker(),
        metadata_extractor=FinanceMetadataExtractor([]),
        classifier=Classifier(),
        settings=FinanceAgentSettings(),
    )
    return build_graph(
        Classifier(), finance_agent=agent, checkpointer=checkpointer or MemorySaver()
    )


def clarification(identifier, unresolved=None, **extra):
    return call(
        CLARIFICATION,
        {
            "question": "Please supply all missing details",
            "unresolved_inputs": unresolved or ["rate_units"],
            **extra,
        },
        identifier,
    )


def model(responses, answer="Done", status="answered"):
    return Model(
        responses=[*responses, AIMessage(content="Done")],
        final=FinanceAgentResult(status=status, answer=answer),
    )


async def test_agent_directly_calls_calculator_with_rag_and_preserves_its_answer():
    answer = "Interest earns interest (page 9). Total interest: $3,382.26."
    m = model(
        [
            call("finance_rag_query", {"query": "Compound interest"}, "rag"),
            call("calculate_compound_interest", INPUTS, "calc"),
        ],
        answer,
    )
    result = await graph(
        m,
        [
            FinanceDocument(
                content="Interest earns interest", source="handbook", metadata={"page_number": 9}
            )
        ],
    ).ainvoke(
        {"query": "What is compound interest and how much will $10000 earn at 6% over 5 years?"},
        {"configurable": {"thread_id": "mixed"}},
    )
    assert result["answer"] == answer
    assert result["calculation"]["interest_earned"] == "3382.26"
    assert result["calculation"]["applied_defaults"] == {
        "currency": "USD",
        "rate_type": "APR",
        "compounding_frequency": "annual",
    }
    assert result["citations"] == ["Financial Education Handbook, page 9"]


async def test_third_reply_can_complete_direct_calculation():
    m = model(
        [
            clarification("one"),
            clarification("two"),
            clarification("three"),
            call("calculate_compound_interest", INPUTS, "calc"),
        ]
    )
    g, config = graph(m), {"configurable": {"thread_id": "third-valid"}}
    result = await g.ainvoke({"query": "Calculate compound interest"}, config)
    for reply in ("10000", "6 percent", "5 years"):
        result = await g.ainvoke(Command(resume=reply), config)
    assert not result.get("__interrupt__")
    assert result["status"] == "answered"
    assert result["calculation"]["clarification_messages_sent"] == 3
    assert result["calculation"]["interest_earned"] == "3382.26"


async def test_third_invalid_reply_ends_with_unresolved_fields():
    m = model(
        [
            clarification("one"),
            clarification("two"),
            clarification("three"),
            call("calculate_compound_interest", {"principal": "10000"}, "invalid"),
        ]
    )
    g, config = graph(m), {"configurable": {"thread_id": "third-invalid"}}
    await g.ainvoke({"query": "Calculate compound interest"}, config)
    for reply in ("unknown", "unknown", "unknown"):
        result = await g.ainvoke(Command(resume=reply), config)
    assert result["status"] == "insufficient"
    assert result["answer"].startswith("Insufficient details to calculate compound interest")
    assert set(result["calculation"]["unresolved_inputs"]) == {
        "annual_rate",
        "rate_units",
        "duration_years",
    }


async def test_fourth_clarification_and_later_calculation_cannot_reset_budget():
    m = model(
        [clarification(str(i)) for i in range(4)]
        + [call("calculate_compound_interest", INPUTS, "bypass")]
    )
    g, config = graph(m), {"configurable": {"thread_id": "fourth"}}
    await g.ainvoke({"query": "Calculate compound interest"}, config)
    for reply in ("unknown", "unknown", "unknown"):
        result = await g.ainvoke(Command(resume=reply), config)
    assert not result.get("__interrupt__")
    assert result["status"] == "insufficient"
    assert result["calculation"]["clarification_messages_sent"] == 3


async def test_invalid_inputs_return_validation_errors_without_arithmetic():
    result = await graph(model([call("calculate_compound_interest", {}, "invalid")])).ainvoke(
        {"query": "Calculate compound interest"}, {"configurable": {"thread_id": "invalid"}}
    )
    assert result["status"] == "insufficient"
    assert result["calculation"]["code"] == "INVALID_INPUT"
    assert "interest_earned" not in result["calculation"]


async def test_agent_repairs_validation_errors_after_clarification():
    m = model(
        [
            call("calculate_compound_interest", {"principal": "10000"}, "invalid"),
            clarification("clarify"),
            call("calculate_compound_interest", INPUTS, "fixed"),
        ]
    )
    g, config = graph(m), {"configurable": {"thread_id": "repair"}}
    await g.ainvoke({"query": "Calculate compound interest on $10000"}, config)
    result = await g.ainvoke(Command(resume="6% over 5 years"), config)
    assert result["status"] == "answered"
    assert result["calculation"]["interest_earned"] == "3382.26"


async def test_partial_explanation_and_state_reach_api():
    from fastapi.testclient import TestClient

    from finassist.api.query import get_query_graph
    from finassist.main import create_app

    m = model(
        [
            call("finance_rag_query", {"query": "Compound interest"}, "rag"),
            clarification(
                "clarify",
                ["annual_rate", "rate_units", "duration_years"],
                explanation="Interest earns interest (page 9).",
            ),
        ]
    )
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph(
        m,
        [
            FinanceDocument(
                content="Interest earns interest", source="handbook", metadata={"page_number": 9}
            )
        ],
    )
    body = (
        TestClient(app)
        .post("/query", json={"query": "Explain compound interest and growth"})
        .json()
    )
    assert body["status"] == "clarification"
    assert body["answer"] == "Interest earns interest (page 9)."
    assert body["calculation"]["clarification_messages_sent"] == 1
    assert body["citations"] == ["Financial Education Handbook, page 9"]


async def test_unsupported_features_require_agent_to_obtain_agreement():
    m = model(
        [
            call(
                "calculate_compound_interest",
                {**INPUTS, "unsupported_features": ["deposits"]},
                "unsupported",
            ),
            clarification("clarify", ["unsupported: deposits"]),
            call("calculate_compound_interest", INPUTS, "simplified"),
        ]
    )
    g, config = graph(m), {"configurable": {"thread_id": "deposits"}}
    await g.ainvoke({"query": "Calculate compound interest with deposits"}, config)
    result = await g.ainvoke(Command(resume="Ignore deposits"), config)
    assert result["calculation"]["interest_earned"] == "3382.26"


async def test_clarification_survives_sqlite_restart(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path, config = str(tmp_path / "calculator.sqlite"), {"configurable": {"thread_id": "durable"}}
    before = model([clarification("clarify")])
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        result = await graph(before, checkpointer=saver).ainvoke(
            {"query": "Calculate compound interest"}, config
        )
        assert result["__interrupt__"][0].value["calculation"]["clarification_messages_sent"] == 1
    after = model([call("calculate_compound_interest", INPUTS, "calc")])
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        result = await graph(after, checkpointer=saver).ainvoke(
            Command(resume="All details"), config
        )
    assert result["calculation"]["clarification_messages_sent"] == 1
    assert result["calculation"]["interest_earned"] == "3382.26"


async def test_invalid_third_reply_cannot_be_followed_by_invented_valid_inputs():
    m = model(
        [clarification(str(i)) for i in range(3)]
        + [
            call("calculate_compound_interest", {}, "invalid"),
            call("calculate_compound_interest", INPUTS, "invented"),
        ]
    )
    g, config = graph(m), {"configurable": {"thread_id": "invalid-bypass"}}
    await g.ainvoke({"query": "Calculate compound interest"}, config)
    for reply in ("unknown", "unknown", "unknown"):
        result = await g.ainvoke(Command(resume=reply), config)
    assert result["status"] == "insufficient"
    assert "interest_earned" not in result["calculation"]


async def test_changed_task_after_calculation_clarification_is_out_of_scope():
    m = model([clarification("clarify")], "Outside scope", status="out_of_scope")
    g, config = graph(m), {"configurable": {"thread_id": "changed"}}
    await g.ainvoke({"query": "Calculate compound interest"}, config)
    result = await g.ainvoke(Command(resume="What is Apple's current price?"), config)
    assert result["status"] == "out_of_scope"
    assert "new question" in result["answer"]


# Labeled calculation regression cases reuse the existing scripted agent harness.
# Scripted calls exercise execution/input preservation, not LLM selection accuracy.
CALCULATION_CASES = [
    json.loads(line)
    for line in (Path(__file__).resolve().parents[1] / "evals/finance/calculations/cases.jsonl")
    .read_text()
    .splitlines()
]


@pytest.mark.parametrize("case", CALCULATION_CASES, ids=lambda case: case["id"])
async def test_labeled_calculation_tool_inputs_and_results(case):
    m = model([call(case["expected_tool"], case["inputs"], "calculation")])
    result = await graph(m).ainvoke(
        {"query": case["query"]},
        {"configurable": {"thread_id": case["id"]}},
    )
    calculation = result["calculation"]
    for field, expected in case["expected"].items():
        assert calculation[field] == expected
    tool_messages = [msg for msg in m.seen[-1] if isinstance(msg, ToolMessage)]
    assert [msg.name for msg in tool_messages] == [case["expected_tool"]]
    assert tool_messages[0].artifact["calculation_id"] == "calculation"
    if calculation["status"] == "success":
        for field, value in case["inputs"].items():
            expected = f"{Decimal(value):.2f}" if field == "principal" else value
            assert calculation["validated_inputs"][field] == expected
        assert calculation["rounding"]["mode"] == "ROUND_HALF_UP"
        assert result["status"] == "answered"
    else:
        assert "final_balance" not in calculation
        assert "interest_earned" not in calculation
        assert result["status"] != "answered"
