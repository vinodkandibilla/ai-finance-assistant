import pytest
from langchain_core.messages import AIMessage

from finassist.agents.tax_education.schemas import TaxAgentResult
from finassist.api.query import QueryResponse, _finalize_response
from test_finance_autonomous_agent import call
from test_tax_agent_graph import make_agent


@pytest.mark.parametrize("status", ["insufficient", "clarification"])
def test_orphan_question_is_not_exposed_as_resumable(status):
    result = _finalize_response(QueryResponse(
        intent="tax", status=status, answer="",
        clarification_question="Which forms should I include?",
    ), status)
    assert result.status == "insufficient"
    assert result.answer.strip()
    assert result.clarification_question is None
    assert result.thread_id is None


def test_real_interrupt_response_is_preserved():
    response = QueryResponse(
        intent="tax", status="clarification", answer="What is your filing status?",
        clarification_question="What is your filing status?", thread_id="paused-thread",
    )
    assert _finalize_response(response, "clarification") == response


def test_partial_insufficient_answer_is_preserved():
    response = QueryResponse(
        intent="tax", status="insufficient", answer="Wages are covered; sale evidence is missing.",
    )
    assert _finalize_response(response, "insufficient") == response


async def test_tax_completion_clears_stray_question_and_supplies_answer():
    agent, _, _ = make_agent(
        [call("tax_rag_query", {"query": "2026 reporting forms"}, "search"),
         AIMessage(content="Done")],
        TaxAgentResult(status="insufficient", answer="",
                       clarification_question="Which forms should I include?"),
    )
    result = await agent.ainvoke({"query": "Explain 2026 reporting forms."}, {})
    assert result["status"] == "insufficient"
    assert result["answer"].strip()
    assert result["clarification_question"] is None
    assert result["final_result"]["clarification_question"] is None


def test_interrupt_with_empty_parent_answer_preserves_question():
    response = QueryResponse(
        intent="market", status="clarification", answer="",
        clarification_question="Please provide dates in YYYY-MM-DD format.",
        thread_id="paused-market-thread",
    )
    result = _finalize_response(response, None)
    assert result.status == "clarification"
    assert result.thread_id == response.thread_id
    assert result.clarification_question == response.clarification_question
    assert result.answer == response.clarification_question
