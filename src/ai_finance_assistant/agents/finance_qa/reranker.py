from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Protocol

from ai_finance_assistant.agents.finance_qa.schemas import FinanceDocument


class CrossEncoderModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> object: ...


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoderModel:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class FinanceCrossEncoderReranker:
    def __init__(self, model_name: str, model: CrossEncoderModel | None = None) -> None:
        self._model_name = model_name
        self._model = model

    async def rerank(
        self, query: str, documents: list[FinanceDocument], *, limit: int
    ) -> list[FinanceDocument]:
        if not documents:
            return []
        scores = await asyncio.to_thread(self._score, query, documents)
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
        model = self._model or _load_cross_encoder(self._model_name)
        return model.predict([(query, document.content) for document in documents])