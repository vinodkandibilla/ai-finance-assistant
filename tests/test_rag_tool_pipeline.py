import pytest

from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.tools import create_finance_rag_query_tool
from finassist.agents.tax_education.tools import create_tax_rag_query_tool
from finassist.rag.models import RetrievedDocument


@pytest.mark.parametrize("domain", ["finance", "tax"])
async def test_standalone_tool_retrieves_then_reranks_and_cites_only_selected_evidence(domain):
    events = []
    documents = [
        RetrievedDocument(
            content="Low relevance", source="corpus", metadata={"page_number": 1, "title": "First"}
        ),
        RetrievedDocument(
            content="High relevance",
            source="corpus",
            metadata={"page_number": 2, "title": "Second"},
        ),
    ]

    class Backend:
        async def retrieve(self, *args, **kwargs):
            events.append("hybrid retrieval")
            return documents, []

        async def search(self, *args, **kwargs):
            events.append("hybrid retrieval")
            return documents

    class Reranker:
        async def rerank(self, query, candidates, limit):
            events.append("rerank")
            assert query == "Explain interest"
            assert candidates == documents
            assert limit == 1
            return [candidates[1]]

    if domain == "finance":
        tool = create_finance_rag_query_tool(
            Backend(), FinanceMetadataExtractor([]), reranker=Reranker(), rerank_top_n=1
        )
    else:
        tool = create_tax_rag_query_tool(Backend(), reranker=Reranker(), rerank_top_n=1)
    result = await tool.ainvoke({"query": "Explain interest"})
    assert events == ["hybrid retrieval", "rerank"]
    assert result["errors"] == []
    assert [doc["content"] for doc in result["candidates"]] == ["High relevance"]
    assert result["citations"] == (
        ["Financial Education Handbook, page 2"] if domain == "finance" else ["Second"]
    )


@pytest.mark.parametrize("domain", ["finance", "tax"])
async def test_standalone_reranking_failure_does_not_leak_unranked_results(domain):
    documents = [RetrievedDocument(content="Unranked evidence", source="corpus")]

    class Backend:
        async def retrieve(self, *args, **kwargs):
            return documents, []

        async def search(self, *args, **kwargs):
            return documents

    class Reranker:
        async def rerank(self, *args, **kwargs):
            raise RuntimeError("private details")

    tool = (
        create_finance_rag_query_tool(Backend(), FinanceMetadataExtractor([]), reranker=Reranker())
        if domain == "finance"
        else create_tax_rag_query_tool(Backend(), reranker=Reranker())
    )
    result = await tool.ainvoke({"query": "Explain interest"})
    assert result["candidates"] == []
    assert result["citations"] == []
    assert result["errors"]
    assert "private details" not in str(result)
