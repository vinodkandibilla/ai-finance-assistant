from langgraph.types import Command

from ai_finance_assistant.agents.finance_qa.calculator_tools import create_calculation_tools
from test_calculation_agent import INPUTS, clarification, graph, model
from test_finance_autonomous_agent import call


async def test_simple_calculator_runs_directly_without_preparation_or_rag():
    result = await graph(
        model([call("calculate_simple_interest", INPUTS, "calc")], "Total simple interest: $3,000.")
    ).ainvoke(
        {"query": "Calculate simple interest on $10000 at 6% over 5 years"},
        {"configurable": {"thread_id": "simple"}},
    )
    assert result["status"] == "answered"
    assert result["calculation"]["interest_earned"] == "3000.00"
    assert "Total simple interest" in result["answer"]


def test_simple_tool_schema_and_defaults():
    tool = create_calculation_tools(kind="simple")[0]
    assert tool.name == "calculate_simple_interest"
    assert "compounding_frequency" not in tool.args
    assert "APY is unsupported" in tool.description


async def test_comparison_keeps_both_results_and_agent_presentation():
    inputs = {**INPUTS, "principal": "100000", "annual_rate": "10", "duration_years": "2"}
    answer = "Compound: $21,000; simple: $20,000. Compound earns more here."
    result = await graph(
        model(
            [
                call("calculate_simple_interest", inputs, "simple"),
                call("calculate_compound_interest", inputs, "compound"),
            ],
            answer,
        )
    ).ainvoke(
        {"query": "Compare interest for 100K over 2 years at 10%"},
        {"configurable": {"thread_id": "compare"}},
    )
    assert result["answer"] == answer
    values = {a["calculation_type"]: a["interest_earned"] for a in result["calculations"]}
    assert values == {"simple": "20000.00", "compound": "21000.00"}


async def test_simple_clarification_limit():
    m = model([clarification(str(i), calculation_type="simple") for i in range(4)])
    g, config = graph(m), {"configurable": {"thread_id": "simple-limit"}}
    await g.ainvoke({"query": "Calculate simple interest"}, config)
    for reply in ("unknown", "unknown", "unknown"):
        result = await g.ainvoke(Command(resume=reply), config)
    assert result["answer"].startswith("Insufficient details to calculate simple interest")
    assert result["calculation"]["clarification_messages_sent"] == 3


async def test_apy_is_rejected_without_arithmetic():
    result = await graph(
        model([call("calculate_simple_interest", {**INPUTS, "rate_type": "APY"}, "invalid")])
    ).ainvoke({"query": "Simple interest at 6% APY"}, {"configurable": {"thread_id": "apy"}})
    assert result["calculation"]["code"] == "INVALID_INPUT"
    assert "interest_earned" not in result["calculation"]
