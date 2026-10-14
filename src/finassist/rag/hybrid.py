from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

from finassist.core.config import get_settings
from finassist.rag.retriever import QdrantHybridRetriever

logger = logging.getLogger(__name__)


def build_hybrid_vector_store(
    collection_name: str,
    local_path: Path,
    documents: list[Document] | Callable[[], list[Document]] | None = None,
    *,
    client: QdrantClient | None = None,
    embedding_model: str | None = None,
    source_name: str = "documents",
    require_documents: bool = True,
) -> QdrantVectorStore:
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not api_key:
        raise RuntimeError(
            f"Set OPENAI_API_KEY in .env or the environment before ingesting "
            f"{source_name} into Qdrant."
        )

    dense_embeddings = OpenAIEmbeddings(
        model=embedding_model or settings.openai_embedding_model,
        api_key=api_key,
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    local_path.mkdir(parents=True, exist_ok=True)

    active_client = client or QdrantClient(path=str(local_path))
    vector_store = QdrantVectorStore(
        client=active_client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=sparse_embeddings,
        validate_collection_config=active_client.collection_exists(collection_name),
    )
    if active_client.collection_exists(collection_name):
        point_count = active_client.count(collection_name=collection_name).count
        if point_count:
            logger.info("Loading Qdrant collection '%s' with %s points.", collection_name, point_count)
            return vector_store
        logger.warning(
            "Qdrant collection '%s' exists but is empty; indexing %s.",
            collection_name,
            source_name,
        )

    if callable(documents):
        documents = documents()
    if documents is None:
        if require_documents:
            logger.error("No %s were provided to build Qdrant collection '%s'.", source_name, collection_name)
        documents = []
    if not documents:
        logger.error("No %s are available to index into '%s'.", source_name, collection_name)
        return vector_store

    dense_vectors = dense_embeddings.embed_documents([documents[0].page_content])
    if not dense_vectors or not dense_vectors[0]:
        logger.error("Dense embeddings are empty; skipping indexing into '%s'.", collection_name)
        return vector_store

    if not active_client.collection_exists(collection_name):
        logger.info("Creating hybrid Qdrant collection '%s'.", collection_name)
        active_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                QdrantVectorStore.VECTOR_NAME: VectorParams(
                    size=len(dense_vectors[0]),
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                QdrantVectorStore.SPARSE_VECTOR_NAME: SparseVectorParams()
            },
        )
    logger.info("Indexing %s documents into Qdrant collection '%s'.", len(documents), collection_name)
    vector_store.add_documents(documents)
    return vector_store


def get_hybrid_retriever(
    collection_name: str,
    local_path: Path,
    documents: Callable[[], list[Document]],
    *,
    client: QdrantClient | None = None,
    embedding_model: str | None = None,
    source_name: str = "documents",
) -> QdrantHybridRetriever:
    vector_store = build_hybrid_vector_store(
        collection_name,
        local_path,
        documents,
        client=client,
        embedding_model=embedding_model,
        source_name=source_name,
    )
    return QdrantHybridRetriever(vector_store)