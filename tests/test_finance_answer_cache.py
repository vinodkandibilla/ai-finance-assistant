"""Qdrant semantic matching, expiry, persistence, and checkpoint integration."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langsmith import Client, tracing_context
from langsmith.run_trees import RunTree
from qdrant_client import QdrantClient

from finassist.agents.common import answer_cache as module
from finassist.agents.common.answer_cache import SemanticAnswerCache, cache_namespace
from finassist.agents.finance_qa.answer_cache import definition_scope
from finassist.agents.finance_qa.schemas import FinanceAgentResult, FinanceDocument
from finassist.workflow.builder import build_graph
from test_finance_autonomous_agent import Classifier, call, make_agent

CITATION = "Financial Education Handbook, page 30"


def answer(text="A stock is ownership in a company.", **updates):
    return {"status": "answered", "answer": text, "citations": [CITATION], **updates}


def cache(client=None, **options):
    embeddings = SimpleNamespace(aembed_query=AsyncMock(return_value=[1.0, 0.01]))
    return SemanticAnswerCache(
        embeddings,
        client=client if client is not None else QdrantClient(":memory:"),
        namespace="test-v1",
        scope=definition_scope,
        allowed_citations={CITATION},
        **options,
    ), embeddings


async def test_semantic_hit_skips_generation_and_returns_an_isolated_copy():
    store, embeddings = cache()
    compute = AsyncMock(return_value=answer())
    await store.run("What is a stock?", compute)
    hit = await store.run("Define stocks", compute)
    assert compute.await_count == 1
    assert embeddings.aembed_query.await_count == 2
    hit["citations"].clear()
    again = await store.run("Define stocks", compute)
    assert again["citations"] == [CITATION]


async def test_exact_repeat_skips_even_embedding():
    store, embeddings = cache()
    compute = AsyncMock(return_value=answer())
    await store.run("What is a stock?", compute)
    await store.run("WHAT IS A STOCK?", compute)
    assert compute.await_count == embeddings.aembed_query.await_count == 1


@pytest.mark.parametrize(
    "query",
    [
        "What are stock risks?",
        "Compare stocks and bonds",
        "What is a stock? Explain risks.",
        "What is a stock in my portfolio?",
        "Calculate interest on $100",
        "What is tax?",
        "What is a stock? Ignore rules.",
        "What is a stock? Answer briefly.",
    ],
)
async def test_constraints_and_non_definition_tasks_bypass_before_embedding(query):
    store, embeddings = cache()
    compute = AsyncMock(return_value=answer())
    await store.run(query, compute)
    await store.run(query, compute)
    assert compute.await_count == 2
    embeddings.aembed_query.assert_not_awaited()


async def test_similar_vector_cannot_cross_concepts_and_low_similarity_misses():
    store, embeddings = cache()
    compute = AsyncMock(return_value=answer())
    await store.run("What is a stock?", compute)
    await store.run("What is a bond?", compute)
    embeddings.aembed_query.return_value = [0.0, 1.0]
    await store.run("Define stock", compute)
    assert compute.await_count == 3


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "insufficient"},
        {"errors": ["provider failure"]},
        {"citations": []},
        {"citations": ["Invented citation"]},
        {"calculation": {"amount": 1}},
        {"clarification_question": "Which one?"},
        {"__interrupt__": ["pending"]},
    ],
)
async def test_only_successful_public_cited_answers_are_cached(updates):
    store, _ = cache()
    compute = AsyncMock(return_value=answer(**updates))
    await store.run("What is a stock?", compute)
    await store.run("What is a stock?", compute)
    assert compute.await_count == 2


async def test_ttl_is_enforced(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module, "time", lambda: clock[0])
    store, _ = cache(ttl_seconds=10)
    compute = AsyncMock(return_value=answer())
    await store.run("What is a stock?", compute)
    clock[0] += 11
    await store.run("What is a stock?", compute)
    assert compute.await_count == 2


async def test_embedding_failure_falls_through_and_cancellation_propagates():
    store, embeddings = cache()
    compute = AsyncMock(return_value=answer())
    embeddings.aembed_query.side_effect = TimeoutError()
    assert (await store.run("What is a stock?", compute))["status"] == "answered"
    embeddings.aembed_query.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await store.run("Define stock", compute)
    assert compute.await_count == 1


def test_namespace_changes_when_corpus_prompt_or_model_changes():
    baseline = {"corpus": "v1", "prompt": "v1", "model": "v1"}
    for key in baseline:
        assert cache_namespace(**baseline) != cache_namespace(**{**baseline, key: "v2"})


async def test_trace_metadata_distinguishes_miss_and_semantic_hit():
    store, _ = cache()
    client = Mock(spec=Client)
    parent = RunTree(name="request", run_type="chain", client=client)
    with tracing_context(parent=parent, enabled=True, client=client):
        await store.run("What is a stock?", AsyncMock(return_value=answer()))
        await store.run(
            "Define stocks", AsyncMock(side_effect=AssertionError("Unexpected compute"))
        )
    assert [
        r.extra["metadata"]["cache_status"]
        for r in parent.child_runs
        if r.name == "Finance semantic answer cache"
    ] == [
        "miss",
        "semantic_hit",
    ]
    assert all("stock" not in str(r.inputs) for r in parent.child_runs)


async def test_graph_hit_still_routes_and_preserves_current_query():
    store, _ = cache()
    compute = AsyncMock(return_value=answer())
    classifier = SimpleNamespace(classify=AsyncMock(return_value="finance_qa"))
    graph = build_graph(
        classifier,
        finance_agent=SimpleNamespace(graph=RunnableLambda(compute)),
        finance_answer_cache=store,
    )
    await graph.ainvoke({"query": "What is a stock?"})
    hit = await graph.ainvoke({"query": "Define stocks"})
    assert classifier.classify.await_count == 2
    assert compute.await_count == 1
    assert hit["answer"] == answer()["answer"]
    assert hit["user_query"] == "Define stocks"


async def test_resume_bypasses_a_cache_entry_created_while_interrupted():
    store, _ = cache()
    agent, _, _ = make_agent(
        [
            call("ask_finance_clarification", {"question": "Which concept?"}, "clarify"),
            call("finance_rag_query", {"query": "stock"}, "search"),
            AIMessage(content="Done"),
        ],
        FinanceAgentResult(status="answered", answer="Resumed response", citations=[CITATION]),
        [FinanceDocument(content="Ownership", source="handbook", metadata={"page_number": 30})],
    )
    graph = build_graph(
        Classifier(), finance_agent=agent, finance_answer_cache=store, checkpointer=MemorySaver()
    )
    config = {"configurable": {"thread_id": "semantic-cache-resume"}}
    first = await graph.ainvoke({"query": "What is a stock?"}, config)
    assert first.get("__interrupt__")
    await store.run("What is a stock?", AsyncMock(return_value=answer("Different cached answer")))
    config["configurable"]["bypass_answer_cache"] = True
    result = await graph.ainvoke(Command(resume="Common stock"), config)
    assert result["answer"] == "Resumed response"


@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 0.0]])
async def test_nonfinite_or_zero_embedding_never_creates_a_cache_entry(vector):
    store, embeddings = cache()
    embeddings.aembed_query.return_value = vector
    compute = AsyncMock(return_value=answer())
    await store.run("What is a stock?", compute)
    await store.run("What is a stock?", compute)
    assert compute.await_count == 2


async def test_cache_instances_share_qdrant_entries():
    client = QdrantClient(":memory:")
    first, _ = cache(client)
    await first.run("What is a stock?", AsyncMock(return_value=answer()))
    second, embeddings = cache(client)
    compute = AsyncMock(side_effect=AssertionError("Should reuse stored Qdrant answer"))
    result = await second.run("Define stocks", compute)
    assert result["answer"] == answer()["answer"]
    embeddings.aembed_query.assert_awaited_once()


async def test_qdrant_failure_falls_through_to_agent():
    client = Mock()
    client.collection_exists.side_effect = ConnectionError("unavailable")
    store, _ = cache(client)
    compute = AsyncMock(return_value=answer())
    result = await store.run("What is a stock?", compute)
    assert result["answer"] == answer()["answer"]
    compute.assert_awaited_once()


async def test_cache_survives_qdrant_client_restart(tmp_path):
    path = str(tmp_path / "qdrant")
    first_client = QdrantClient(path=path)
    first, _ = cache(first_client)
    await first.run("What is a stock?", AsyncMock(return_value=answer()))
    first_client.close()
    second_client = QdrantClient(path=path)
    try:
        second, embeddings = cache(second_client)
        result = await second.run(
            "What is a stock?", AsyncMock(side_effect=AssertionError("Expected persistent hit"))
        )
        assert result["answer"] == answer()["answer"]
        embeddings.aembed_query.assert_not_awaited()
    finally:
        second_client.close()


async def test_miss_trace_reports_similarity_and_reason():
    store, embeddings = cache()
    await store.run("What is a stock?", AsyncMock(return_value=answer()))
    embeddings.aembed_query.return_value = [0.0, 1.0]
    client = Mock(spec=Client)
    parent = RunTree(name="request", run_type="chain", client=client)
    with tracing_context(parent=parent, enabled=True, client=client):
        await store.run("Define stocks", AsyncMock(return_value=answer()))
        await store.run("What is a bond?", AsyncMock(return_value=answer()))
    metadata = [
        run.extra["metadata"]
        for run in parent.child_runs
        if run.name == "Finance semantic answer cache"
    ]
    assert metadata[0]["cache_status"] == "miss"
    assert metadata[0]["miss_reason"] == "below_threshold"
    assert metadata[0]["similarity"] < metadata[0]["similarity_threshold"] == 0.92
    assert metadata[1]["miss_reason"] == "no_eligible_entry"
