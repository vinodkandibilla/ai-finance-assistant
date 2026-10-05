from langchain_core.documents import Document

from ai_finance_assistant.rag.retriever import QdrantRetriever


class FakeVectorStore:
    async def asimilarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        assert query == "budget"
        assert k == 2
        return [(Document(page_content="Monthly budget", metadata={"source": "guide.pdf"}), 0.9)]


async def test_qdrant_retriever_maps_documents_and_scores() -> None:
    retriever = QdrantRetriever(FakeVectorStore())

    results = await retriever.search("budget", limit=2)

    assert len(results) == 1
    assert results[0].content == "Monthly budget"
    assert results[0].source == "guide.pdf"
    assert results[0].score == 0.9
