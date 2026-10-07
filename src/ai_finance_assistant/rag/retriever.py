from typing import Protocol

from langchain_core.documents import Document
from qdrant_client.models import Filter

from ai_finance_assistant.rag.models import RetrievedDocument


class VectorSearchBackend(Protocol):
    async def asimilarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Filter | dict | None = None,
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

    @property
    def vector_store(self) -> VectorSearchBackend:
        return self._vector_store

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: Filter | dict | None = None,
    ) -> list[RetrievedDocument]:
        if metadata_filter is None:
            matches = await self._vector_store.asimilarity_search_with_score(query, k=limit)
        else:
            matches = await self._vector_store.asimilarity_search_with_score(
                query,
                k=limit,
                filter=metadata_filter,
            )
        return [
            RetrievedDocument(
                content=document.page_content,
                source=str(document.metadata.get("source", "unknown")),
                metadata={key: str(value) for key, value in document.metadata.items()},
                score=score,
            )
            for document, score in matches
        ]
