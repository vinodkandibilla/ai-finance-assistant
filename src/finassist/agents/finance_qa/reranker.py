from __future__ import annotations

import asyncio
import logging
import math
import platform
from functools import lru_cache
from threading import RLock
from typing import Protocol

from langsmith import trace

from finassist.agents.finance_qa.schemas import FinanceDocument

_MODEL_LOCK = RLock()
logger = logging.getLogger(__name__)


class CrossEncoderModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> object: ...


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoderModel:
    from sentence_transformers import CrossEncoder

    # Avoid auto-selecting Apple's MPS backend for the server's threaded inference.
    return CrossEncoder(model_name, device="cpu" if platform.system() == "Darwin" else None)


class FinanceCrossEncoderReranker:
    def __init__(self, model_name: str, model: CrossEncoderModel | None = None) -> None:
        self._model_name = model_name
        self._model = model

    async def rerank(
        self, query: str, documents: list[FinanceDocument], *, limit: int
    ) -> list[FinanceDocument]:
        if not documents:
            return []
        with trace(
            "Rerank evidence",
            run_type="retriever",
            inputs={"candidate_count": len(documents), "limit": limit},
            metadata={"model": self._model_name},
        ) as span:
            scores = [
                float(score) for score in await asyncio.to_thread(self._score, query, documents)
            ]
            if len(scores) != len(documents):
                raise ValueError("Reranker score count does not match candidate count")
            if not all(math.isfinite(score) for score in scores):
                logger.warning("reranker_nonfinite_scores using_original_retrieval_order")
                span.add_metadata({"ranking_status": "fallback_nonfinite_scores"})
                return [
                    document.model_copy(
                        update={
                            "score": document.score
                            if document.score is not None and math.isfinite(document.score)
                            else None,
                            "rank": rank,
                            "provenance": [*document.provenance, "rerank_fallback_nonfinite"],
                        }
                    )
                    for rank, document in enumerate(documents[:limit], start=1)
                ]
            span.add_metadata({"ranking_status": "reranked"})
        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            document.model_copy(update={"score": float(score), "rank": rank})
            for rank, (document, score) in enumerate(ranked[:limit], start=1)
        ]

    def _score(self, query: str, documents: list[FinanceDocument]) -> object:
        # Finance and tax share a cached model. lru_cache alone does not prevent
        # duplicate concurrent loads; guard loading and native inference together.
        # Keep the lock in the worker: cancellation must not release it while
        # native prediction is still running in that worker.
        with trace("Wait for reranker lock", run_type="chain"):
            _MODEL_LOCK.acquire()
        try:
            with trace(
                "Get or load reranker model", run_type="chain", metadata={"model": self._model_name}
            ):
                model = self._model or _load_cross_encoder(self._model_name)
            with trace(
                "Score evidence pairs", run_type="chain", inputs={"pair_count": len(documents)}
            ):
                return model.predict(
                    [(query, document.content) for document in documents], show_progress_bar=False
                )
        finally:
            _MODEL_LOCK.release()
