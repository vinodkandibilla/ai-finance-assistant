"""Shared native reranker access is serialized across finance and tax requests."""

import asyncio
import sys
import time
from types import SimpleNamespace

import pytest

from finassist.agents.finance_qa import reranker as runtime


@pytest.mark.parametrize("system,device", [("Darwin", "cpu"), ("Linux", None)])
def test_device_selection(monkeypatch, system, device):
    calls = []
    model = object()

    def create(name, **kwargs):
        calls.append((name, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=create))
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    runtime._load_cross_encoder.cache_clear()
    try:
        assert runtime._load_cross_encoder("test-model") is model
        assert calls == [("test-model", {"device": device})]
    finally:
        runtime._load_cross_encoder.cache_clear()


@pytest.mark.asyncio
async def test_shared_model_load_and_prediction_are_serialized(monkeypatch):
    active = 0
    peak = 0
    loads = 0

    class Model:
        def predict(self, pairs, **kwargs):
            nonlocal active, peak
            assert kwargs == {"show_progress_bar": False}
            active += 1
            peak = max(peak, active)
            time.sleep(0.02)
            active -= 1
            return [1.0]

    def create(name, **kwargs):
        nonlocal loads
        loads += 1
        time.sleep(0.02)
        return Model()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=create))
    runtime._load_cross_encoder.cache_clear()
    try:
        first = runtime.FinanceCrossEncoderReranker("shared-test-model")
        second = runtime.FinanceCrossEncoderReranker("shared-test-model")
        documents = [SimpleNamespace(content="Evidence")]
        await asyncio.gather(
            asyncio.to_thread(first._score, "Question", documents),
            asyncio.to_thread(second._score, "Question", documents),
        )
        assert loads == 1
        assert peak == 1
    finally:
        runtime._load_cross_encoder.cache_clear()


async def test_reranker_exposes_finite_scores_in_rank_order():
    from finassist.agents.finance_qa.schemas import FinanceDocument

    reranker = runtime.FinanceCrossEncoderReranker(
        "test",
        model=SimpleNamespace(predict=lambda pairs, **kwargs: [-1.5, 3.2]),
    )
    documents = [FinanceDocument(content=text, source="handbook") for text in ["first", "second"]]
    result = await reranker.rerank("ETF", documents, limit=2)
    assert [(d.content, d.score, d.rank) for d in result] == [
        ("second", 3.2, 1),
        ("first", -1.5, 2),
    ]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_scores_preserve_retrieval_order_and_identify_fallback(invalid, caplog):
    import json

    from finassist.agents.finance_qa.schemas import FinanceDocument

    reranker = runtime.FinanceCrossEncoderReranker(
        "test",
        model=SimpleNamespace(predict=lambda pairs, **kwargs: [invalid, 4.0]),
    )
    documents = [
        FinanceDocument(content="first", source="handbook", score=0.02),
        FinanceDocument(content="second", source="handbook", score=invalid),
    ]
    result = await reranker.rerank("ETF", documents, limit=2)
    assert [(d.content, d.score, d.rank) for d in result] == [
        ("first", 0.02, 1),
        ("second", None, 2),
    ]
    assert all("rerank_fallback_nonfinite" in d.provenance for d in result)
    json.dumps([d.model_dump() for d in result], allow_nan=False)
    assert "reranker_nonfinite_scores" in caplog.text
