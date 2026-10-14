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
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.schemas import FinanceDocument
from finassist.rag.models import RetrievedDocument
from finassist.rag.retriever import QdrantHybridSearch

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
        self._hybrid_search = QdrantHybridSearch(
            qdrant_client,
            self._collection_name,
            embeddings,
            sparse_embeddings,
        )
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
        needs_indexing = True
        if qdrant_client.collection_exists(collection_name):
            sparse_config = (
                qdrant_client.get_collection(collection_name).config.params.sparse_vectors or {}
            )
            if QdrantVectorStore.SPARSE_VECTOR_NAME not in sparse_config:
                raise ValueError(
                    f"Finance collection '{collection_name}' must have sparse vectors configured."
                )
            needs_indexing = qdrant_client.count(collection_name, exact=True).count == 0
            if not needs_indexing:
                logger.info("Reusing finance hybrid collection '%s'.", collection_name)
        else:
            dense_vectors = embeddings.embed_documents([documents[0].page_content])
            if not dense_vectors or not dense_vectors[0]:
                logger.error(
                    "Finance embeddings are empty; skipping indexing into '%s'.", collection_name
                )
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
        if needs_indexing:
            logger.info("Indexing %d handbook chunks into '%s'.", len(documents), collection_name)
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
                points = await self._hybrid_search.search(
                    dense_queries,
                    query,
                    dense_limit=self._settings.dense_top_k,
                    sparse_limit=self._settings.sparse_top_k,
                    candidate_limit=self._settings.rrf_candidates,
                    rrf_k=self._settings.rrf_k,
                    timeout_seconds=self._settings.retrieval_timeout_seconds,
                    query_filter=metadata_filter,
                )
            documents: list[RetrievedDocument] = []
            for rank, point in enumerate(points, start=1):
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
