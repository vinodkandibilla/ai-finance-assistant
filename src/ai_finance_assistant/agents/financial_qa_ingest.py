from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.rag.retriever import QdrantRetriever

logger = logging.getLogger(__name__)

DEFAULT_METADATA = {
    "document_title": "Financial Foundations for Students",
    "jurisdiction": "US",
}
COLLECTION_NAME = "financial_foundations_hybrid"
LOCAL_QDRANT_PATH = Path(__file__).resolve().parents[3] / ".qdrant"
PDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "rag"
    / "finance-qa"
    / "Financial_Education_Handbook_integrated.pdf"
)


def _extract_structure_metadata(document: Document) -> dict[str, str | int | None]:
    page_content = document.page_content or ""
    page_number = document.metadata.get("page_number")

    chapter_match = re.search(r"(?i)\bchapter\s+(\d+(?:\.\d+)?)", page_content)
    section_match = re.search(
        r"(?i)(?:^|\s)(?:section\s+)?(\d+\.\d+(?:\.\d+)?)\b",
        page_content,
    )

    if chapter_match is None:
        chapter_fallback = re.search(r"(?i)\b(\d+\.\d+)\b", page_content)
        chapter = chapter_fallback.group(1).strip() if chapter_fallback else None
    else:
        chapter = chapter_match.group(1).strip()

    if section_match is not None:
        section = section_match.group(1).strip()
    else:
        section = None

    return {
        "chapter": chapter,
        "section": section,
        "page_number": int(page_number) if page_number is not None else None,
    }


def load_pdf_elements(pdf_path: Path = PDF_PATH) -> list[Document]:
    loader = UnstructuredPDFLoader(str(pdf_path), mode="elements")
    documents = loader.load()

    for document in documents:
        structure = _extract_structure_metadata(document)
        document.metadata = {**DEFAULT_METADATA, **document.metadata, **structure}

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\nChapter ", "\n\n", "\n"],
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        structure = _extract_structure_metadata(chunk)
        chunk.metadata = {**DEFAULT_METADATA, **chunk.metadata, **structure}
        chunk.metadata["chunk_id"] = f"chunk-{index:04d}"

    return chunks


def build_hybrid_vector_store(documents: list[Document] | None = None) -> QdrantVectorStore:
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env or the environment before ingesting the PDF into Qdrant.")

    dense_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant_client = QdrantClient(path=str(LOCAL_QDRANT_PATH))
    point_count = 0
    try:
        if qdrant_client.collection_exists(COLLECTION_NAME):
            point_count = qdrant_client.count(collection_name=COLLECTION_NAME).count
            if point_count > 0:
                logger.info("Loading existing Qdrant collection '%s' from local disk.", COLLECTION_NAME)
            else:
                logger.warning(
                    "Qdrant collection '%s' exists but contains no vectors; rebuilding it to generate embeddings.",
                    COLLECTION_NAME,
                )
                qdrant_client.delete_collection(COLLECTION_NAME)
    finally:
        qdrant_client.close()

    if point_count > 0:
        return QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=dense_embeddings,
            path=str(LOCAL_QDRANT_PATH),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=sparse_embeddings,
        )

    logger.info(
        "Qdrant collection '%s' not found or empty at '%s'; building embeddings and creating it from scratch.",
        COLLECTION_NAME,
        str(LOCAL_QDRANT_PATH),
    )
    if documents is None:
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist and no documents were provided to build it."
        )

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=dense_embeddings,
        collection_name=COLLECTION_NAME,
        path=str(LOCAL_QDRANT_PATH),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=sparse_embeddings,
    )
    point_count = vector_store.client.count(collection_name=COLLECTION_NAME).count
    logger.info(
        "Qdrant collection '%s' contains %s dense and %s sparse embedding vectors.",
        COLLECTION_NAME,
        point_count,
        point_count,
    )
    return vector_store


def validate_hybrid_search(vector_store: QdrantVectorStore, query: str, k: int = 3) -> list[Document]:
    return vector_store.similarity_search(query=query, k=k)


@lru_cache
def get_finance_qa_retriever() -> QdrantRetriever:
    qdrant_client = QdrantClient(path=str(LOCAL_QDRANT_PATH))
    try:
        collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)
        point_count = (
            qdrant_client.count(collection_name=COLLECTION_NAME).count
            if collection_exists
            else 0
        )
    finally:
        qdrant_client.close()

    if collection_exists and point_count > 0:
        logger.info(
            "Using existing Qdrant index '%s' with %s points.",
            COLLECTION_NAME,
            point_count,
        )
        return QdrantRetriever(build_hybrid_vector_store())

    logger.info(
        "Qdrant index '%s' is missing or empty; creating it from the PDF source.",
        COLLECTION_NAME,
    )
    raw_documents = load_pdf_elements(PDF_PATH)
    chunks = chunk_documents(raw_documents)
    vector_store = build_hybrid_vector_store(chunks)
    return QdrantRetriever(vector_store)


def main() -> None:
    raw_documents = load_pdf_elements(PDF_PATH)
    chunks = chunk_documents(raw_documents)
    vector_store = build_hybrid_vector_store(chunks)

    sample_query = (
        "What are the most important financial foundations students should know about "
        "saving, budgeting, and long-term planning?"
    )
    results = validate_hybrid_search(vector_store, sample_query, k=3)

    logger.info("Loaded %s PDF elements and created %s chunks.", len(raw_documents), len(chunks))
    logger.info("Stored chunks in Qdrant collection: %s", COLLECTION_NAME)
    logger.info("Top hybrid search results:")
    for index, result in enumerate(results, start=1):
        logger.info("Result %s: %s...", index, result.page_content[:220])
        logger.info("Metadata: %s", result.metadata)


if __name__ == "__main__":
    main()
