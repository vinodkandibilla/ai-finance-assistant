"""Shared Qdrant-backed semantic answers with domain-specific eligibility."""

import asyncio
import copy
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from time import time
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from langsmith import trace
from qdrant_client import QdrantClient, models


class QueryEmbeddings(Protocol):
    async def aembed_query(self, text: str) -> list[float]: ...


def cache_namespace(**versions: Any) -> str:
    return hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()


class SemanticAnswerCache:
    def __init__(
        self,
        embeddings: QueryEmbeddings,
        client: QdrantClient,
        *,
        namespace: str,
        scope: Callable[[str], tuple[str, str] | None],
        intent: str = "finance_qa",
        label: str = "Finance",
        allowed_citations: set[str],
        collection: str = "finance_answer_cache",
        ttl_seconds: float = 3600,
        similarity: float = 0.92,
        embedding_timeout_seconds: float = 2,
    ) -> None:
        if not 0 < similarity <= 1:
            raise ValueError("Invalid cache similarity")
        if any(not math.isfinite(v) or v <= 0 for v in [ttl_seconds, embedding_timeout_seconds]):
            raise ValueError("Invalid cache timeout")
        self._scope = scope
        self._intent = intent
        self._label = label
        self._embeddings = embeddings
        self._namespace = namespace
        self._allowed_citations = frozenset(allowed_citations)
        self._client = client
        self._collection = collection
        self._collection_lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._similarity = similarity
        self._embedding_timeout = embedding_timeout_seconds

    async def run(
        self,
        query: str,
        compute: Callable[[], Awaitable[dict[str, Any]]],
        *,
        bypass: bool = False,
    ) -> dict[str, Any]:
        scope = None if bypass else self._scope(query)
        vector = None
        with trace(
            f"{self._label} semantic answer cache",
            run_type="chain",
            inputs={"eligible": scope is not None},
            metadata={"namespace": self._namespace, "similarity_threshold": self._similarity},
        ) as span:
            if scope is None:
                span.add_metadata({"cache_status": "bypass"})
            else:
                key, concept = scope
                identifier = str(uuid5(NAMESPACE_URL, self._namespace + ":" + key))
                try:
                    exists = await asyncio.to_thread(
                        self._client.collection_exists, self._collection
                    )
                    if exists:
                        hits = await asyncio.to_thread(
                            self._client.retrieve,
                            self._collection,
                            [identifier],
                            with_payload=True,
                        )
                        if hits and (hits[0].payload or {}).get("expires", 0) > time():
                            span.add_metadata({"cache_status": "exact_hit", "similarity": 1.0})
                            return copy.deepcopy(dict((hits[0].payload or {})["result"]))
                    async with asyncio.timeout(self._embedding_timeout):
                        with trace(
                            "Embed cache query",
                            run_type="embedding",
                            inputs={"query_length": len(query)},
                        ):
                            vector = await self._embeddings.aembed_query(query)
                    if not vector or not all(math.isfinite(v) for v in vector) or not any(vector):
                        raise ValueError("Invalid cache embedding")
                    async with self._collection_lock:
                        if not await asyncio.to_thread(
                            self._client.collection_exists, self._collection
                        ):
                            try:
                                await asyncio.to_thread(
                                    self._client.create_collection,
                                    self._collection,
                                    vectors_config=models.VectorParams(
                                        size=len(vector), distance=models.Distance.COSINE
                                    ),
                                )
                            except Exception:
                                if not await asyncio.to_thread(
                                    self._client.collection_exists, self._collection
                                ):
                                    raise
                    response = await asyncio.to_thread(
                        self._client.query_points,
                        self._collection,
                        query=vector,
                        query_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="namespace", match=models.MatchValue(value=self._namespace)
                                ),
                                models.FieldCondition(
                                    key="concept", match=models.MatchValue(value=concept)
                                ),
                                models.FieldCondition(key="expires", range=models.Range(gt=time())),
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                    )
                    if response.points:
                        hit = response.points[0]
                        span.add_metadata({"similarity": hit.score})
                        if hit.score >= self._similarity:
                            span.add_metadata({"cache_status": "semantic_hit"})
                            return copy.deepcopy(dict((hit.payload or {})["result"]))
                        span.add_metadata(
                            {"cache_status": "miss", "miss_reason": "below_threshold"}
                        )
                    else:
                        span.add_metadata(
                            {"cache_status": "miss", "miss_reason": "no_eligible_entry"}
                        )
                except Exception as error:
                    vector = None
                    span.add_metadata(
                        {"cache_status": "cache_error", "error_category": type(error).__name__}
                    )
        result = await compute()
        citations = result.get("citations") or []
        if (
            scope
            and vector is not None
            and result.get("status") == "answered"
            and str(result.get("answer", "")).strip()
            and citations
            and set(citations) <= self._allowed_citations
            and not result.get("errors")
            and not result.get("clarification_question")
            and not any(result.get(k) for k in ("calculation", "calculations", "__interrupt__"))
        ):
            # Store only the public outcome, never graph state or user context.
            answer = {
                k: copy.deepcopy(result[k])
                for k in ("answer", "citations", "status", "errors", "clarification_question")
                if k in result
            }
            answer.update(intent=self._intent, final_result=copy.deepcopy(answer))
            with trace(f"Store {self._label.lower()} cached answer", run_type="chain") as span:
                try:
                    # Expired entries are never served; reclaim them on successful writes.
                    await asyncio.to_thread(
                        self._client.delete,
                        self._collection,
                        models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="expires", range=models.Range(lte=time())
                                    )
                                ]
                            )
                        ),
                        wait=True,
                    )
                    await asyncio.to_thread(
                        self._client.upsert,
                        self._collection,
                        [
                            models.PointStruct(
                                id=identifier,
                                vector=vector,
                                payload={
                                    "namespace": self._namespace,
                                    "query": scope[0],
                                    "concept": scope[1],
                                    "expires": time() + self._ttl,
                                    "result": answer,
                                },
                            )
                        ],
                        wait=True,
                    )
                    span.add_metadata({"cache_status": "stored"})
                except Exception as error:
                    span.add_metadata(
                        {"cache_status": "store_error", "error_category": type(error).__name__}
                    )
        return result
