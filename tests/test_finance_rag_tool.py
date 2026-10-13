import pytest

from ai_finance_assistant.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from ai_finance_assistant.agents.finance_qa.schemas import FinanceDocument
from ai_finance_assistant.agents.finance_qa.tools import create_finance_rag_query_tool


class Reranker:
    async def rerank(self, query, documents, limit):
        return documents[:limit]


@pytest.mark.asyncio
async def test_tool_preserves_search_plan_filters_and_evidence():
    class Retriever:
        async def retrieve(self, dense_queries, sparse_keywords, *, metadata_filter):
            assert dense_queries == ["ETF NAV", "ETF market price"]
            assert sparse_keywords == ["ETF", "NAV"]
            assert metadata_filter.must[0].key == "metadata.topic"
            assert metadata_filter.must[0].match.value == "ETF"
            return [
                FinanceDocument(
                    content="ETF evidence", source="handbook.pdf", metadata={"page_number": 9}
                )
            ], []

    tool = create_finance_rag_query_tool(
        Retriever(), FinanceMetadataExtractor([{"topic": "ETF"}]), reranker=Reranker()
    )
    result = await tool.ainvoke(
        {
            "query": "Explain ETF prices",
            "dense_queries": ["ETF NAV", "ETF market price"],
            "sparse_keywords": ["ETF", "NAV"],
        }
    )
    assert tool.name == "finance_rag_query"
    assert result["metadata_filters"] == {"topic": "ETF"}
    assert result["candidates"][0]["metadata"]["page_number"] == 9
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_tool_defaults_to_question_and_preserves_retrieval_errors():
    class Retriever:
        async def retrieve(self, dense_queries, sparse_keywords, *, metadata_filter):
            assert dense_queries == ["What is compound interest?"]
            assert sparse_keywords == ["What is compound interest?"]
            assert metadata_filter is None
            return [], ["hybrid retrieval failed: TimeoutError"]

    tool = create_finance_rag_query_tool(
        Retriever(), FinanceMetadataExtractor([]), reranker=Reranker()
    )
    result = await tool.ainvoke({"query": "What is compound interest?"})
    assert result["candidates"] == []
    assert result["errors"] == ["hybrid retrieval failed: TimeoutError"]
