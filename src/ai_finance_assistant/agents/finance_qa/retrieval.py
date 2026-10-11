from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    Modifier,
    Prefetch,
    Rrf,
    RrfQuery,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ai_finance_assistant.agents.finance_qa.config import FinanceAgentSettings
from ai_finance_assistant.agents.finance_qa.schemas import FinanceDocument

logger = logging.getLogger(__name__)


class FinanceHybridRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        qdrant_client: QdrantClient,
        embeddings: object,
        sparse_embeddings: object,
        settings: FinanceAgentSettings,
    ) -> None:
        self._settings = settings
        self._qdrant_client = qdrant_client
        self._collection_name = settings.finance_collection
        self._index_ready = qdrant_client.collection_exists(settings.finance_collection)
        if not self._index_ready:
            logger.error(
                "Finance collection '%s' is unavailable; retrieval will return no chunks.",
                settings.finance_collection,
            )
        self._embeddings = embeddings
        self._sparse_embeddings = sparse_embeddings
        self._semaphore = asyncio.Semaphore(settings.max_parallel_searches)

    @classmethod
    def create(
        cls,
        documents: Sequence[Document],
        *,
        qdrant_client: QdrantClient,
        api_key: str,
        embedding_model: str,
        settings: FinanceAgentSettings,
    ) -> FinanceHybridRetriever:
        embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        collection_name = settings.finance_collection
        if not documents:
            logger.error("No finance documents are available to index into '%s'.", collection_name)
            return cls(
                documents,
                qdrant_client=qdrant_client,
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
                settings=settings,
            )
        if qdrant_client.collection_exists(collection_name):
            sparse_config = (
                qdrant_client.get_collection(collection_name).config.params.sparse_vectors
                or {}
            )
            if QdrantVectorStore.SPARSE_VECTOR_NAME not in sparse_config:
                raise ValueError(
                    f"Finance collection '{collection_name}' must have sparse vectors configured."
                )
            logger.info("Reusing finance hybrid collection '%s'.", collection_name)
        else:
            dense_vectors = embeddings.embed_documents([documents[0].page_content])
            if not dense_vectors or not dense_vectors[0]:
                logger.error("Finance embeddings are empty; skipping indexing into '%s'.", collection_name)
                return cls(
                    documents,
                    qdrant_client=qdrant_client,
                    embeddings=embeddings,
                    sparse_embeddings=sparse_embeddings,
                    settings=settings,
                )
            vector_size = len(dense_vectors[0])
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    QdrantVectorStore.VECTOR_NAME: VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    QdrantVectorStore.SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
                },
            )
            vector_store = QdrantVectorStore(
                embedding=embeddings,
                collection_name=collection_name,
                client=qdrant_client,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_embedding=sparse_embeddings,
            )
            vector_store.add_documents(list(documents))
        return cls(
            documents,
            qdrant_client=qdrant_client,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            settings=settings,
        )

    async def retrieve(
        self,
        dense_queries: list[str],
        sparse_keywords: list[str],
        *,
        metadata_filter: object | None = None,
    ) -> tuple[list[FinanceDocument], list[str]]:
        if not getattr(self, "_index_ready", True):
            return [], []
        query = " ".join(sparse_keywords).strip()
        try:
            async with self._semaphore:
                async with asyncio.timeout(self._settings.retrieval_timeout_seconds):
                    dense_vectors = await asyncio.gather(
                        *(self._embeddings.aembed_query(value) for value in dense_queries)
                    )
                    prefetch = [
                        Prefetch(
                            query=vector,
                            using=QdrantVectorStore.VECTOR_NAME,
                            limit=self._settings.dense_top_k,
                            filter=metadata_filter,
                        )
                        for vector in dense_vectors
                        if vector
                    ]
                    if query:
                        sparse_vector = await asyncio.to_thread(
                            self._sparse_embeddings.embed_query, query
                        )
                        if sparse_vector.indices:
                            prefetch.append(
                                Prefetch(
                                    query=SparseVector(
                                        indices=sparse_vector.indices,
                                        values=sparse_vector.values,
                                    ),
                                    using=QdrantVectorStore.SPARSE_VECTOR_NAME,
                                    limit=self._settings.sparse_top_k,
                                    filter=metadata_filter,
                                )
                            )
                    if not prefetch:
                        return [], []
                    response = await asyncio.to_thread(
                        self._qdrant_client.query_points,
                        collection_name=self._collection_name,
                        prefetch=prefetch,
                        query=RrfQuery(rrf=Rrf(k=self._settings.rrf_k)),
                        query_filter=metadata_filter,
                        limit=self._settings.rrf_candidates,
                        with_payload=True,
                    )
            documents: list[FinanceDocument] = []
            for rank, point in enumerate(response.points, start=1):
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                content = payload.get("page_content") or payload.get("content") or ""
                documents.append(
                    FinanceDocument(
                        doc_id=str(metadata.get("doc_id", point.id)),
                        content=str(content),
                        metadata=metadata,
                        source=str(metadata.get("source", "unknown")),
                        score=float(point.score),
                        rank=rank,
                    )
                )
            return documents, []
        except Exception as error:
            logger.exception("Finance hybrid retrieval with Qdrant RRF failed.")
            return [], [f"hybrid retrieval failed: {type(error).__name__}"]
