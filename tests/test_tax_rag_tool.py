import pytest

from ai_finance_assistant.agents.tax_education.tools import create_tax_rag_query_tool
from ai_finance_assistant.rag.models import RetrievedDocument


class Reranker:
    async def rerank(self, query, documents, limit):
        return documents[:limit]


async def test_tax_tool_preserves_original_constraints_on_reformulation():
    class Retriever:
        async def search(self, query, **kwargs):
            assert query == "Deduction eligibility tax year 2025"
            assert kwargs["dense_queries"] == ["Deduction eligibility"]
            assert kwargs["sparse_query"] == "deduction eligibility"
            conditions = {c.key: c.match for c in kwargs["metadata_filter"].must}
            assert conditions["metadata.tax_year"].value == 2026
            assert "single" in conditions["metadata.filing_status"].any
            return [
                RetrievedDocument(
                    content="Tax evidence",
                    source="tax.jsonl",
                    metadata={"title": "Deductions", "tax_year": 2026},
                )
            ]

    tool = create_tax_rag_query_tool(Retriever(), reranker=Reranker())
    result = await tool.ainvoke(
        {
            "query": "Explain deductions for Single filing status, tax year 2026",
            "search_query": "Deduction eligibility tax year 2025",
            "dense_queries": ["Deduction eligibility"],
            "sparse_keywords": ["deduction", "eligibility"],
        }
    )
    assert tool.name == "tax_rag_query"
    assert result["metadata_filters"]["tax_year"] == 2026
    assert result["errors"] == []
    assert result["candidates"][0]["metadata"]["title"] == "Deductions"
    assert result["errors"] == []


async def test_tax_tool_defaults_to_original_question_and_returns_errors():
    class Retriever:
        async def search(self, query, **kwargs):
            assert query == "What is a standard deduction?"
            assert kwargs["dense_queries"] == [query]
            raise TimeoutError("private provider details")

    result = await create_tax_rag_query_tool(Retriever(), reranker=Reranker()).ainvoke(
        {"query": "What is a standard deduction?"}
    )
    assert result["candidates"] == []
    assert result["errors"] == ["TimeoutError"]


async def test_tax_tool_rejects_empty_original_question():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await create_tax_rag_query_tool(None, reranker=Reranker()).ainvoke({"query": ""})


async def test_bracket_question_does_not_filter_by_unrelated_filing_status_subtopic():
    class Retriever:
        async def search(self, query, **kwargs):
            conditions = {c.key: c.match for c in kwargs["metadata_filter"].must}
            assert conditions["metadata.topic"].value == "tax_brackets"
            assert "mfj" in conditions["metadata.filing_status"].any
            assert "metadata.subtopic" not in conditions
            return []

    result = await create_tax_rag_query_tool(Retriever(), reranker=Reranker()).ainvoke(
        {
            "query": "What will be my tax bracket for the income $250K "
            "with filing status Married filing jointly"
        }
    )
    assert result["errors"] == []
    assert result["metadata_filters"]["topic"] == "tax_brackets"
    assert "subtopic" not in result["metadata_filters"]
