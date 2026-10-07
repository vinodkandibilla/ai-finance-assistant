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

from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.rag.retriever import QdrantRetriever

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
            f"Set OPENAI_API_KEY in .env or the environment before ingesting {source_name} into Qdrant."
        )

    dense_embeddings = OpenAIEmbeddings(
        model=embedding_model or settings.openai_embedding_model,
        api_key=api_key,
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    local_path.mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    inspection_client = client or QdrantClient(path=str(local_path))
    point_count = 0
    try:
        if inspection_client.collection_exists(collection_name):
            point_count = inspection_client.count(collection_name=collection_name).count
            if point_count > 0:
                logger.info("Loading Qdrant collection '%s' from local disk.", collection_name)
            else:
                logger.warning(
                    "Qdrant collection '%s' exists but contains no vectors; rebuilding it to generate embeddings.",
                    collection_name,
                )
                inspection_client.delete_collection(collection_name)
    finally:
        if owns_client:
            inspection_client.close()

    if point_count > 0:
        if client is not None:
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=dense_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_embedding=sparse_embeddings,
            )
        return QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            embedding=dense_embeddings,
            path=str(local_path),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=sparse_embeddings,
        )

    if callable(documents):
        documents = documents()
    if documents is None:
        if require_documents:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' does not exist and no {source_name} were provided to build it."
            )
        documents = []

    logger.info(
        "Qdrant collection '%s' not found or empty at '%s'; building embeddings and creating it from scratch.",
        collection_name,
        str(local_path),
    )
    if client is None:
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=dense_embeddings,
            collection_name=collection_name,
            path=str(local_path),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=sparse_embeddings,
        )
    else:
        if not documents:
            raise RuntimeError(f"No {source_name} were provided to build Qdrant collection '{collection_name}'.")
        dense_vectors = dense_embeddings.embed_documents([documents[0].page_content])
        client.create_collection(
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
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=sparse_embeddings,
        )
        vector_store.add_documents(documents)

    point_count = vector_store.client.count(collection_name=collection_name).count
    logger.info(
        "Qdrant collection '%s' contains %s dense and %s sparse embedding vectors.",
        collection_name,
        point_count,
        point_count,
    )
    return vector_store


def get_hybrid_retriever(
    collection_name: str,
    local_path: Path,
    documents: Callable[[], list[Document]],
    *,
    client: QdrantClient | None = None,
    embedding_model: str | None = None,
    source_name: str = "documents",
) -> QdrantRetriever:
    vector_store = build_hybrid_vector_store(
        collection_name,
        local_path,
        documents,
        client=client,
        embedding_model=embedding_model,
        source_name=source_name,
    )
    return QdrantRetriever(vector_store)