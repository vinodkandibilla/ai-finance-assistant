from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue

from ai_finance_assistant.rag.retriever import QdrantHybridRetriever, QdrantRetriever


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


async def test_qdrant_retriever_returns_empty_chunks_when_search_fails() -> None:
    class UnavailableVectorStore:
        async def asimilarity_search_with_score(self, query: str, *, k: int = 4):
            raise RuntimeError("collection is not initialized")

    retriever = QdrantRetriever(UnavailableVectorStore())

    assert await retriever.search("budget") == []


async def test_qdrant_hybrid_retriever_uses_fused_points_and_keeps_metadata() -> None:
    from types import SimpleNamespace

    class DenseEmbeddings:
        async def aembed_query(self, query: str) -> list[float]:
            assert query == "MFJ tax brackets"
            return [0.1, 0.2]

    class SparseEmbeddings:
        def embed_query(self, query: str):
            assert query == "MFJ tax brackets"
            return SimpleNamespace(indices=[2], values=[0.5])

    class Client:
        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="point-1",
                        score=0.9,
                        payload={
                            "page_content": "The 22% MFJ bracket applies above $100,800.",
                            "metadata": {
                                "title": "2026 MFJ Tax Brackets",
                                "chunk_id": "irs2026_045_mfj_brackets-0001",
                                "source": "irs-corpus.jsonl",
                            },
                        },
                    )
                ]
            )

    client = Client()
    vector_store = SimpleNamespace(
        client=client,
        collection_name="tax_education_hybrid",
        embeddings=DenseEmbeddings(),
        sparse_embeddings=SparseEmbeddings(),
    )
    retriever = QdrantHybridRetriever(vector_store)
    query_filter = Filter(
        must=[FieldCondition(key="metadata.topic", match=MatchValue(value="tax_brackets"))]
    )

    results = await retriever.search(
        "MFJ tax brackets",
        limit=3,
        metadata_filter=query_filter,
    )

    assert len(results) == 1
    assert results[0].content == "The 22% MFJ bracket applies above $100,800."
    assert results[0].metadata["title"] == "2026 MFJ Tax Brackets"
    assert results[0].metadata["chunk_id"] == "irs2026_045_mfj_brackets-0001"
    assert results[0].source == "irs-corpus.jsonl"
    assert results[0].score == 0.9
    assert len(client.kwargs["prefetch"]) == 2
    assert all(item.filter == query_filter for item in client.kwargs["prefetch"])
    assert client.kwargs["query_filter"] == query_filter
    assert client.kwargs["limit"] == 3
