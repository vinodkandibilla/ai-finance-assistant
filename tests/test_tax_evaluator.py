import pytest

from ai_finance_assistant.agents.finance_qa.schemas import ContextEvaluation as FinanceEvaluation
from ai_finance_assistant.agents.tax_education.evaluator import TaxContextEvaluator
from ai_finance_assistant.rag.evaluation import ContextEvaluation
from ai_finance_assistant.rag.models import RetrievedDocument


class FakeLLM:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is ContextEvaluation
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.result


def test_finance_evaluation_import_remains_compatible():
    assert FinanceEvaluation is ContextEvaluation


@pytest.mark.asyncio
@pytest.mark.parametrize("year", [2025, None])
async def test_tax_evaluator_rejects_wrong_or_missing_year_without_model_call(year):
    llm = FakeLLM()
    document = RetrievedDocument(
        content="Deduction rule.", source="tax.jsonl", metadata={"tax_year": year}
    )
    result = await TaxContextEvaluator(llm).evaluate(
        "Explain deductions for tax year 2026", [document]
    )
    assert not result.sufficient
    assert "2026" in result.reason
    assert not llm.calls


@pytest.mark.asyncio
async def test_tax_evaluator_passes_question_constraints_and_actual_evidence():
    llm = FakeLLM(ContextEvaluation(sufficient=True, reason="Covered"))
    document = RetrievedDocument(
        content="Tax evidence.", source="tax.jsonl", metadata={"tax_year": 2026}
    )
    result = await TaxContextEvaluator(llm).evaluate(
        "Explain deductions for tax year 2026", [document]
    )
    assert result.sufficient
    assert "Tax evidence." in llm.calls[0][1].content
    assert "Return sufficient" in llm.calls[0][0].content


@pytest.mark.asyncio
async def test_tax_evaluator_fails_closed_for_malformed_output_and_empty_context():
    llm = FakeLLM({"invalid": True})
    evaluator = TaxContextEvaluator(llm)
    document = RetrievedDocument(content="Tax evidence.", source="tax.jsonl")
    result = await evaluator.evaluate("What is a deduction?", [document])
    assert not result.sufficient
    empty = await evaluator.evaluate("What is a deduction?", [])
    assert not empty.sufficient
    assert len(llm.calls) == 1
