"""Shared Qdrant hybrid search and document conversion for Finance/Tax RAG."""

import asyncio
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore
from langsmith import trace
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, Prefetch, Rrf, RrfQuery, ScoredPoint, SparseVector

from finassist.rag.models import RetrievedDocument

logger = logging.getLogger(__name__)


class QdrantHybridSearch:
    """Embed queries and let Qdrant fuse dense/sparse candidate rankings."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: OpenAIEmbeddings,
        sparse_embeddings: FastEmbedSparse,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._sparse_embeddings = sparse_embeddings

    async def search(
        self,
        dense_queries: list[str],
        sparse_query: str,
        *,
        dense_limit: int,
        sparse_limit: int,
        candidate_limit: int,
        rrf_k: int,
        timeout_seconds: float = 15.0,
        query_filter: Filter | dict | None = None,
    ) -> list[ScoredPoint]:
        try:
            async with asyncio.timeout(timeout_seconds):
                with trace(
                    "Embed dense queries",
                    run_type="embedding",
                    inputs={"query_count": len(dense_queries)},
                ):
                    dense_vectors = await asyncio.gather(
                        *(self._embeddings.aembed_query(query) for query in dense_queries)
                    )
                prefetch = [
                    Prefetch(
                        query=vector,
                        using=QdrantVectorStore.VECTOR_NAME,
                        limit=dense_limit,
                        filter=query_filter,
                    )
                    for vector in dense_vectors
                    if vector
                ]
                if sparse_query.strip():
                    with trace(
                        "Encode sparse query",
                        run_type="embedding",
                        inputs={"query_length": len(sparse_query)},
                    ):
                        sparse_vector = await asyncio.to_thread(
                            self._sparse_embeddings.embed_query, sparse_query
                        )
                    if sparse_vector.indices:
                        prefetch.append(
                            Prefetch(
                                query=SparseVector(
                                    indices=sparse_vector.indices,
                                    values=sparse_vector.values,
                                ),
                                using=QdrantVectorStore.SPARSE_VECTOR_NAME,
                                limit=sparse_limit,
                                filter=query_filter,
                            )
                        )
                if not prefetch:
                    return []
                with trace(
                    "Qdrant hybrid search and RRF",
                    run_type="retriever",
                    inputs={
                        "collection": self._collection_name,
                        "prefetch_count": len(prefetch),
                        "candidate_limit": candidate_limit,
                        "filtered": query_filter is not None,
                    },
                ) as span:
                    response = await asyncio.to_thread(
                        self._client.query_points,
                        collection_name=self._collection_name,
                        prefetch=prefetch,
                        query=RrfQuery(rrf=Rrf(k=rrf_k)),
                        query_filter=query_filter,
                        limit=candidate_limit,
                        with_payload=True,
                    )
                    span.add_outputs({"result_count": len(response.points)})
            return response.points
        except Exception:
            logger.exception(
                "Qdrant hybrid retrieval failed for collection '%s'.",
                self._collection_name,
            )
            raise


class QdrantHybridRetriever:
    """Adapt hybrid search results to the application evidence model."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        *,
        dense_limit: int = 10,
        sparse_limit: int = 10,
        candidate_limit: int = 20,
        rrf_k: int = 60,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._search = QdrantHybridSearch(
            vector_store.client,
            vector_store.collection_name,
            vector_store.embeddings,
            vector_store.sparse_embeddings,
        )
        self._dense_limit = dense_limit
        self._sparse_limit = sparse_limit
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k
        self._timeout_seconds = timeout_seconds

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        dense_queries: list[str] | None = None,
        sparse_query: str | None = None,
        metadata_filter: Filter | dict | None = None,
    ) -> list[RetrievedDocument]:
        try:
            points = await self._search.search(
                dense_queries or [query],
                query if sparse_query is None else sparse_query,
                dense_limit=self._dense_limit,
                sparse_limit=self._sparse_limit,
                candidate_limit=min(self._candidate_limit, limit),
                rrf_k=self._rrf_k,
                timeout_seconds=self._timeout_seconds,
                query_filter=metadata_filter,
            )
        except Exception:
            return []
        results: list[RetrievedDocument] = []
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            results.append(
                RetrievedDocument(
                    content=str(payload.get("page_content") or payload.get("content") or ""),
                    source=str(metadata.get("source", "unknown")),
                    metadata=metadata,
                    score=float(point.score),
                    doc_id=str(metadata.get("doc_id", point.id)),
                    rank=len(results) + 1,
                )
            )
        return results
