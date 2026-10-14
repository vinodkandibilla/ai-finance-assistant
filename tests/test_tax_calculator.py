from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from finassist.agents.tax_education.calculator_tools import CALCULATOR
from finassist.agents.tax_education.schemas import TaxAgentResult
from finassist.agents.tax_education.tax_calculation import (
    THRESHOLDS,
    calculate_federal_income_tax,
)
from finassist.workflow.builder import build_graph
from test_finance_autonomous_agent import call
from test_tax_agent_graph import Classifier, make_agent

INPUTS = {"taxable_income": "150000", "filing_status": "single", "income_basis": "taxable_ordinary"}


def test_sample_progressive_tax_and_defaults():
    result = calculate_federal_income_tax(INPUTS)
    assert result["status"] == "success"
    assert result["estimated_federal_income_tax"] == "28598.00"
    assert result["marginal_rate_percent"] == 24
    assert result["effective_rate_percent"] == "19.07"
    assert [b["tax"] for b in result["bracket_breakdown"]] == [
        "1240.00",
        "4560.00",
        "12166.00",
        "10632.00",
    ]
    assert result["applied_defaults"] == {"tax_year": 2026, "currency": "USD"}
    assert "payments" in " ".join(result["assumptions"])


@pytest.mark.parametrize(
    "status,expected",
    [
        ("single", "28598.00"),
        ("mfj", "22424.00"),
        ("mfs", "28598.00"),
        ("hoh", "26787.00"),
        ("qss", "22424.00"),
    ],
)
def test_all_filing_statuses(status, expected):
    result = calculate_federal_income_tax({**INPUTS, "filing_status": status})
    assert result["estimated_federal_income_tax"] == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("taxable_income", "-1"),
        ("taxable_income", "NaN"),
        ("taxable_income", "Infinity"),
        ("taxable_income", "150K"),
        ("taxable_income", "1e19"),
        ("taxable_income", 150000),
        ("filing_status", "unknown"),
        ("income_basis", "gross"),
        ("tax_year", 2025),
        ("currency", "EUR"),
    ],
)
def test_invalid_or_unsupported_inputs_never_return_tax(field, value):
    result = calculate_federal_income_tax({**INPUTS, field: value})
    assert result["status"] == "error"
    assert "estimated_federal_income_tax" not in result


def test_missing_inputs_zero_and_rounding():
    assert len(calculate_federal_income_tax({})["errors"]) == 3
    result = calculate_federal_income_tax({**INPUTS, "taxable_income": "0"})
    assert result["estimated_federal_income_tax"] == "0.00"
    assert result["effective_rate_percent"] == "0.00"
    result = calculate_federal_income_tax({**INPUTS, "taxable_income": "1.055"})
    assert result["validated_inputs"]["taxable_income"] == "1.06"
    assert result["estimated_federal_income_tax"] == "0.11"


@pytest.mark.parametrize("status", THRESHOLDS)
def test_tax_is_continuous_across_every_threshold(status):
    for threshold in THRESHOLDS[status]:
        taxes = [
            Decimal(
                calculate_federal_income_tax(
                    {**INPUTS, "filing_status": status, "taxable_income": str(threshold + offset)}
                )["estimated_federal_income_tax"]
            )
            for offset in (-1, 0, 1)
        ]
        assert taxes[0] <= taxes[1] <= taxes[2]
        assert taxes[2] - taxes[0] < 1
    assert (
        calculate_federal_income_tax(
            {**INPUTS, "filing_status": status, "taxable_income": "1000000"}
        )["marginal_rate_percent"]
        == 37
    )


async def test_agent_calls_rag_and_direct_calculator_for_sample():
    answer = "2026: your marginal rate is 24%; estimated federal ordinary tax is $28,598."
    agent, retriever, _ = make_agent(
        [
            call("tax_rag_query", {"query": "2026 single brackets"}, "rag"),
            call(CALCULATOR, INPUTS, "calc"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer=answer),
    )
    result = await agent.graph.ainvoke(
        {
            "query": "What tax brackets apply to $150,000 of taxable income filing single, "
            "and how much tax will I owe?"
        }
    )
    assert len(retriever.calls) == 1
    assert result["answer"] == answer
    assert result["calculation"]["estimated_federal_income_tax"] == "28598.00"
    assert result["citations"]


async def test_pure_calculation_does_not_require_rag():
    agent, retriever, _ = make_agent(
        [call(CALCULATOR, INPUTS, "calc"), AIMessage(content="Done")],
        TaxAgentResult(status="answered", answer="$28,598 before credits and payments."),
    )
    result = await agent.graph.ainvoke({"query": "Calculate federal ordinary tax on 150K single"})
    assert result["status"] == "answered"
    assert not retriever.calls


async def test_missing_status_resumes_direct_calculation():
    agent, _, _ = make_agent(
        [
            call("ask_tax_clarification", {"question": "What is your filing status?"}, "clarify"),
            call(CALCULATOR, INPUTS, "calc"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Estimated tax: $28,598."),
    )
    graph = build_graph(Classifier(), tax_agent=agent, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "tax-calculator-resume"}}
    result = await graph.ainvoke({"query": "Tax owed for $150000 taxable income"}, config)
    assert result.get("__interrupt__")
    result = await graph.ainvoke(Command(resume="Single"), config)
    assert result["calculation"]["estimated_federal_income_tax"] == "28598.00"


async def test_calculator_error_cannot_support_answer_and_can_be_repaired():
    agent, _, _ = make_agent(
        [
            call(CALCULATOR, {**INPUTS, "income_basis": "gross"}, "invalid"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Invented amount"),
    )
    result = await agent.graph.ainvoke({"query": "Tax on gross salary"})
    assert result["status"] == "insufficient"
    assert "estimated_federal_income_tax" not in result["calculation"]
    agent, _, _ = make_agent(
        [
            call(CALCULATOR, {}, "invalid"),
            call(CALCULATOR, INPUTS, "fixed"),
            AIMessage(content="Done"),
        ],
        TaxAgentResult(status="answered", answer="Corrected estimate"),
    )
    result = await agent.graph.ainvoke({"query": "Tax estimate"})
    assert result["status"] == "answered"
    assert result["calculation"]["estimated_federal_income_tax"] == "28598.00"


async def test_api_exposes_structured_tax_calculation():
    from fastapi.testclient import TestClient

    from finassist.api.query import get_query_graph
    from finassist.main import create_app

    agent, _, _ = make_agent(
        [call(CALCULATOR, INPUTS, "calc"), AIMessage(content="Done")],
        TaxAgentResult(status="answered", answer="Estimated tax: $28,598."),
    )
    graph = build_graph(Classifier(), tax_agent=agent, checkpointer=MemorySaver())
    app = create_app()
    app.dependency_overrides[get_query_graph] = lambda: graph
    response = TestClient(app).post(
        "/query", json={"query": "Tax on $150000 single taxable income"}
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "tax"
    assert response.json()["calculation"]["estimated_federal_income_tax"] == "28598.00"
