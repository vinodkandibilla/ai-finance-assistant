import pytest

from ai_finance_assistant.agents.finance_qa.schemas import QueryAnalysis
from ai_finance_assistant.agents.tax_education.query_analyzer import TaxQueryAnalyzer
from ai_finance_assistant.rag.tax_education import TaxHybridRetriever, build_tax_metadata_filter


class StructuredLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.messages = []

    def with_structured_output(self, schema):
        assert schema is QueryAnalysis
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_tax_analysis_uses_tax_prompt_and_preserves_explicit_terms():
    query = "Explain 2026 Form 1040 deductions for married filing jointly."
    llm = StructuredLLM(
        QueryAnalysis(
            needs_clarification=False,
            detected_intents=["tax"],
            dense_sub_queries=["2026 Form 1040 deductions for married filing jointly"],
            sparse_keywords=["Form 1040", "married filing jointly"],
        )
    )
    result = await TaxQueryAnalyzer(llm).analyze(query)
    assert "IRS tax-education" in llm.messages[0].content
    assert "Out of scope: retirement accounts" not in llm.messages[0].content
    assert llm.messages[1].content == query
    assert result.detected_intents == ["tax"]
    assert "2026" in result.sparse_keywords
    assert "1040" in result.sparse_keywords
    assert "married filing jointly" in result.sparse_keywords


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "malformed"])
async def test_tax_analysis_failure_keeps_original_query_and_tax_intent(failure):
    query = "Explain 2026 Form 1040 deductions."
    llm = StructuredLLM(
        result={"unexpected": "output"},
        error=RuntimeError("unavailable") if failure == "exception" else None,
    )
    result = await TaxQueryAnalyzer(llm).analyze(query)
    assert result.detected_intents == ["tax"]
    assert result.dense_sub_queries == [query]
    assert "2026" in result.sparse_keywords
    assert "1040" in result.sparse_keywords


@pytest.mark.asyncio
async def test_tax_decomposition_preserves_filter_and_original_question():
    query = "Compare deductions for married filing jointly in 2026."
    llm = StructuredLLM(
        QueryAnalysis(
            needs_clarification=False,
            detected_intents=["tax"],
            dense_sub_queries=["2026 joint standard deduction", "2026 joint itemized deduction"],
            sparse_keywords=["deduction"],
        )
    )

    class Backend:
        async def search(self, received_query, **kwargs):
            self.query = received_query
            self.kwargs = kwargs
            return []

    backend = Backend()
    metadata_filter = build_tax_metadata_filter({"tax_year": 2026})
    retriever = TaxHybridRetriever(backend, TaxQueryAnalyzer(llm))
    await retriever.search(query, limit=3, metadata_filter=metadata_filter)
    assert backend.query == query
    assert backend.kwargs["metadata_filter"] is metadata_filter
    assert backend.kwargs["limit"] == 3
    assert backend.kwargs["dense_queries"][0] == query
    assert len(backend.kwargs["dense_queries"]) == 3
    assert "2026" in backend.kwargs["sparse_query"]


@pytest.mark.asyncio
async def test_tax_definition_retains_original_query_before_model_expansion():
    query = "What is a standard deduction?"
    llm = StructuredLLM(
        QueryAnalysis(
            needs_clarification=False,
            detected_intents=["tax"],
            dense_sub_queries=["2023 standard deduction amounts"],
        )
    )
    result = await TaxQueryAnalyzer(llm).analyze(query)
    assert result.dense_sub_queries == [query]
    assert "do not invent a tax year" in llm.messages[0].content


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What will be my tax bracket", "taxable income and filing status"),
        ("What will be my tax bracket in 2026 filing single", "taxable income?"),
        ("What will be my tax bracket with income 130K", "filing status"),
    ],
)
async def test_personal_bracket_missing_inputs_clarifies_before_model(query, expected):
    llm = StructuredLLM(error=AssertionError("Should not need model"))
    result = await TaxQueryAnalyzer(llm).analyze(query)
    assert result.needs_clarification
    assert expected in result.clarification_question
    assert llm.messages == []


@pytest.mark.parametrize(
    "query",
    [
        "What will be my tax bracket with income 130K filing single",
        "Explain tax brackets",
    ],
)
async def test_complete_bracket_or_concept_question_proceeds(query):
    llm = StructuredLLM(QueryAnalysis(needs_clarification=False, detected_intents=["tax"]))
    result = await TaxQueryAnalyzer(llm).analyze(query)
    assert not result.needs_clarification
    assert llm.messages
