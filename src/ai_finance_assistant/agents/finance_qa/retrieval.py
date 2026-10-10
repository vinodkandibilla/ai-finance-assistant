from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

from ai_finance_assistant.agents.finance_qa.config import FinanceAgentSettings
from ai_finance_assistant.agents.finance_qa.schemas import FinanceDocument

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*", text.casefold())


class FinanceHybridRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        qdrant_client: QdrantClient,
        embeddings: object,
        settings: FinanceAgentSettings,
    ) -> None:
        self._documents = list(documents)
        self._settings = settings
        self._bm25 = BM25Okapi([tokenize(document.page_content) for document in self._documents])
        self._dense_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=settings.finance_collection,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
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
        if not documents:
            raise ValueError("No in-scope finance documents were provided for indexing.")
        embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
        if not qdrant_client.collection_exists(settings.finance_collection):
            # Factory methods create their own client; use the existing client
            # directly so local storage is not reopened and kwargs do not leak
            # into the underlying HTTP client.
            vector_size = len(embeddings.embed_documents([documents[0].page_content])[0])
            qdrant_client.create_collection(
                collection_name=settings.finance_collection,
                vectors_config={
                    QdrantVectorStore.VECTOR_NAME: VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    )
                },
            )
            vector_store = QdrantVectorStore(
                embedding=embeddings,
                collection_name=settings.finance_collection,
                client=qdrant_client,
                retrieval_mode=RetrievalMode.DENSE,
            )
            vector_store.add_documents(list(documents))
        else:
            logger.info("Reusing finance dense collection '%s'.", settings.finance_collection)
        return cls(
            documents,
            qdrant_client=qdrant_client,
            embeddings=embeddings,
            settings=settings,
        )

    async def retrieve(
        self,
        dense_queries: list[str],
        sparse_keywords: list[str],
        *,
        metadata_filter: object | None = None,
    ) -> tuple[list[list[FinanceDocument]], list[FinanceDocument], list[str]]:
        dense_tasks = [self._dense_search(query, metadata_filter) for query in dense_queries]
        sparse_task = asyncio.to_thread(self._sparse_search, sparse_keywords, metadata_filter)
        results = await asyncio.gather(*dense_tasks, sparse_task, return_exceptions=True)
        ranked_lists: list[list[FinanceDocument]] = []
        errors: list[str] = []
        for result in results[:-1]:
            if isinstance(result, Exception):
                errors.append(f"dense retrieval failed: {type(result).__name__}")
                continue
            ranked_lists.append(result)
        sparse_result = results[-1]
        sparse_list: list[FinanceDocument] = []
        if isinstance(sparse_result, Exception):
            errors.append(f"sparse retrieval failed: {type(sparse_result).__name__}")
        else:
            sparse_list = sparse_result
        if errors:
            logger.warning("Finance retrieval degraded: %s", "; ".join(errors))
        return ranked_lists, sparse_list, errors

    async def _dense_search(self, query: str, metadata_filter: object | None) -> list[FinanceDocument]:
        async with self._semaphore:
            async with asyncio.timeout(self._settings.retrieval_timeout_seconds):
                matches = await self._dense_store.asimilarity_search_with_score(
                    query,
                    k=self._settings.dense_top_k,
                    filter=metadata_filter,
                )
        return [
            self._from_document(document, score, rank)
            for rank, (document, score) in enumerate(matches, 1)
        ]

    def _sparse_search(
        self, keywords: list[str], metadata_filter: object | None
    ) -> list[FinanceDocument]:
        query_tokens = tokenize(" ".join(keywords))
        if not query_tokens or not self._documents:
            return []
        scores = self._bm25.get_scores(query_tokens)
        scored = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        results: list[FinanceDocument] = []
        for index, score in scored:
            document = self._documents[index]
            if metadata_filter is not None and not _matches_filter(document.metadata, metadata_filter):
                continue
            results.append(self._from_document(document, float(score), len(results) + 1))
            if len(results) >= self._settings.sparse_top_k:
                break
        return results

    @staticmethod
    def _from_document(document: Document, score: float, rank: int) -> FinanceDocument:
        metadata = document.metadata
        return FinanceDocument(
            doc_id=str(metadata["doc_id"]),
            content=document.page_content,
            metadata=metadata,
            source=str(metadata["source"]),
            score=float(score),
            rank=rank,
        )


def _matches_filter(metadata: dict[str, object], qdrant_filter: object) -> bool:
    for condition in getattr(qdrant_filter, "must", []) or []:
        key = str(condition.key).removeprefix("metadata.")
        expected = getattr(condition.match, "value", None)
        if expected is not None and str(metadata.get(key)) != str(expected):
            return False
    return True
