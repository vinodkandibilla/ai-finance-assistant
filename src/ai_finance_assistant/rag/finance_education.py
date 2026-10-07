from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from ai_finance_assistant.rag.hybrid import (
    build_hybrid_vector_store,
    get_hybrid_retriever,
)
from ai_finance_assistant.rag.retriever import QdrantRetriever

logger = logging.getLogger(__name__)

DEFAULT_METADATA = {
    "document_title": "Financial Foundations for Students",
    "jurisdiction": "US",
}
FINANCE_COLLECTION_NAME = "financial_foundations_hybrid"
LOCAL_QDRANT_PATH = Path(__file__).resolve().parents[3] / ".qdrant"
FINANCE_CORPUS_PATH = (
    Path(__file__).resolve().parent
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

    section = section_match.group(1).strip() if section_match is not None else None
    return {
        "chapter": chapter,
        "section": section,
        "page_number": int(page_number) if page_number is not None else None,
    }


def load_finance_documents(pdf_path: Path = FINANCE_CORPUS_PATH) -> list[Document]:
    """Load finance education PDF elements and attach normalized structure metadata."""
    loader = UnstructuredPDFLoader(str(pdf_path), mode="elements")
    documents = loader.load()

    for document in documents:
        structure = _extract_structure_metadata(document)
        document.metadata = {**DEFAULT_METADATA, **document.metadata, **structure}

    return documents


def chunk_finance_documents(documents: list[Document]) -> list[Document]:
    """Split finance documents into retrieval chunks while preserving source metadata."""
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


def build_finance_hybrid_vector_store(
    documents: list[Document] | None = None,
) -> QdrantVectorStore:
    return build_hybrid_vector_store(
        FINANCE_COLLECTION_NAME,
        LOCAL_QDRANT_PATH,
        documents,
        embedding_model="text-embedding-3-small",
        source_name="the finance PDF",
    )


@lru_cache
def _get_finance_retriever_without_shared_client() -> QdrantRetriever:
    return get_hybrid_retriever(
        FINANCE_COLLECTION_NAME,
        LOCAL_QDRANT_PATH,
        lambda: chunk_finance_documents(load_finance_documents(FINANCE_CORPUS_PATH)),
        embedding_model="text-embedding-3-small",
        source_name="the finance PDF",
    )


def get_finance_retriever(client: QdrantClient | None = None) -> QdrantRetriever:
    if client is None:
        return _get_finance_retriever_without_shared_client()
    return get_hybrid_retriever(
        FINANCE_COLLECTION_NAME,
        LOCAL_QDRANT_PATH,
        lambda: chunk_finance_documents(load_finance_documents(FINANCE_CORPUS_PATH)),
        client=client,
        embedding_model="text-embedding-3-small",
        source_name="the finance PDF",
    )


def validate_finance_hybrid_search(
    vector_store: QdrantVectorStore,
    query: str,
    limit: int = 3,
) -> list[Document]:
    return vector_store.similarity_search(query=query, k=limit)


def main() -> None:
    raw_documents = load_finance_documents(FINANCE_CORPUS_PATH)
    chunks = chunk_finance_documents(raw_documents)
    vector_store = build_finance_hybrid_vector_store(chunks)
    sample_query = (
        "What are the most important financial foundations students should know about "
        "saving, budgeting, and long-term planning?"
    )
    results = validate_finance_hybrid_search(vector_store, sample_query, limit=3)

    logger.info("Loaded %s PDF elements and created %s chunks.", len(raw_documents), len(chunks))
    logger.info("Stored chunks in Qdrant collection: %s", FINANCE_COLLECTION_NAME)
    logger.info("Top hybrid search results:")
    for index, result in enumerate(results, start=1):
        logger.info("Result %s: %s...", index, result.page_content[:220])
        logger.info("Metadata: %s", result.metadata)


if __name__ == "__main__":
    main()