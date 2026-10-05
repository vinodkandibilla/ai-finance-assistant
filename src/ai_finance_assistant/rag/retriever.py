from typing import Protocol

from langchain_core.documents import Document

from ai_finance_assistant.rag.models import RetrievedDocument


class VectorSearchBackend(Protocol):
    async def asimilarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
    ) -> list[tuple[Document, float]]: ...


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedDocument]: ...


class QdrantRetriever:
    def __init__(self, vector_store: VectorSearchBackend) -> None:
        self._vector_store = vector_store

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedDocument]:
        matches = await self._vector_store.asimilarity_search_with_score(query, k=limit)
        return [
            RetrievedDocument(
                content=document.page_content,
                source=str(document.metadata.get("source", "unknown")),
                metadata={key: str(value) for key, value in document.metadata.items()},
                score=score,
            )
            for document, score in matches
        ]
