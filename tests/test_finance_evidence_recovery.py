"""Recovery actions preserve constraints and enforce request-local budgets."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.evidence_recovery import (
    BASELINE,
    EXPAND,
    TARGET,
    create_evidence_recovery,
)
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.schemas import FinanceDocument


class Retriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, dense, sparse, **kwargs):
        self.calls.append((dense, sparse, kwargs))
        documents = [
            FinanceDocument(
                doc_id=str(i),
                content=f"Evidence {i}",
                source="handbook",
                metadata={"page_number": i},
            )
            for i in range(1, 12 if "profile" in kwargs else 6)
        ]
        return documents, []


def setup(reranker=None, **settings):
    retriever = Retriever()
    execute = create_evidence_recovery(
        retriever,
        FinanceMetadataExtractor([{"topic": "ETF"}]),
        reranker,
        FinanceAgentSettings(**settings),
    )
    return retriever, execute, {"messages": [HumanMessage(content="Explain ETF fees and trading")]}


async def invoke(execute, state, action, query="ETF", **kwargs):
    call_id = str(len(state["messages"]))
    state["messages"].append(
        AIMessage(
            content="", tool_calls=[{"name": action, "args": {"query": query}, "id": call_id}]
        )
    )
    payload, artifact = await execute(action, query, state, **kwargs)
    state["messages"].append(
        ToolMessage(
            content="evidence",
            artifact=artifact,
            name=action,
            tool_call_id=call_id,
        )
    )
    return payload


async def test_expansion_reuses_plan_and_returns_only_novel_evidence():
    retriever, execute, state = setup()
    baseline = await invoke(execute, state, BASELINE, dense=["ETF fees"], sparse=["fees"])
    expanded = await invoke(
        execute,
        state,
        EXPAND,
        query="ignored unrelated query",
        missing_aspect="ETF trading costs beyond the initial fee explanation",
    )
    assert baseline["new_passages"] == 3
    assert expanded["new_passages"] == 5
    assert expanded["duplicates_skipped"] == 3
    assert retriever.calls[1][:2] == retriever.calls[0][:2]
    assert retriever.calls[1][2]["profile"] == {
        "dense_limit": 20,
        "sparse_limit": 20,
        "candidate_limit": 40,
    }
    assert expanded["remaining_searches"] == 2
    assert expanded["metadata_filters"] == {"topic": "ETF"}
    blocked = await invoke(execute, state, EXPAND, missing_aspect="Trading costs")
    assert "limit" in blocked["errors"][0]
    assert len(retriever.calls) == 2


async def test_targeted_search_preserves_original_metadata_and_requires_gap():
    retriever, execute, state = setup()
    await invoke(execute, state, BASELINE)
    blocked = await invoke(execute, state, TARGET, query="fees")
    assert "missing aspect" in blocked["errors"][0]
    result = await invoke(execute, state, TARGET, query="expense ratios", missing_aspect="fees")
    assert result["metadata_filters"] == {"topic": "ETF"}
    assert retriever.calls[-1][0] == ["expense ratios"]
    assert "profile" not in retriever.calls[-1][2]


async def test_recovery_requires_baseline_and_total_limit_is_shared():
    retriever, execute, state = setup(agent_max_searches=2)
    blocked = await invoke(execute, state, TARGET, missing_aspect="fees")
    assert "baseline" in blocked["errors"][0]
    await invoke(execute, state, BASELINE)
    exhausted = await invoke(execute, state, EXPAND, missing_aspect="Trading costs")
    assert "budget exhausted" in exhausted["errors"][0]
    assert len(retriever.calls) == 1


async def test_evidence_size_budget_and_parallel_retrieval_are_enforced():
    retriever, execute, state = setup(max_evidence_characters=10)
    result = await invoke(execute, state, BASELINE)
    assert result["new_passages"] == 1
    assert result["remaining_characters"] == 0
    blocked = await invoke(execute, state, EXPAND, missing_aspect="Trading costs")
    assert "budget exhausted" in blocked["errors"][0]
    assert len(retriever.calls) == 1


async def test_targeted_action_limit_and_semantic_limit():
    retriever, execute, state = setup(max_targeted_actions=1, max_dense_queries=1)
    await invoke(execute, state, BASELINE)
    await invoke(execute, state, TARGET, missing_aspect="fees")
    blocked = await invoke(execute, state, TARGET, missing_aspect="trading")
    assert "limit" in blocked["errors"][0]
    assert len(retriever.calls) == 2
    retriever, execute, state = setup(max_dense_queries=1)
    blocked = await invoke(execute, state, BASELINE, dense=["a", "b"])
    assert "Semantic query limit" in blocked["errors"][0]
    assert not retriever.calls


async def test_baseline_cannot_bypass_recovery_limits():
    retriever, execute, state = setup()
    await invoke(execute, state, BASELINE)
    blocked = await invoke(execute, state, BASELINE, query="different wording")
    assert "already attempted" in blocked["errors"][0]
    assert len(retriever.calls) == 1


async def test_provider_failure_is_reported_without_fabricating_evidence():
    retriever, execute, state = setup()

    async def fail(*args, **kwargs):
        raise TimeoutError()

    retriever.retrieve = fail
    result = await invoke(execute, state, BASELINE)
    assert result["candidates"] == []
    assert result["citations"] == []
    assert result["errors"] == ["Finance search failed: TimeoutError"]


async def test_expansion_without_specific_gap_does_not_retrieve():
    retriever, execute, state = setup()
    await invoke(execute, state, BASELINE)
    blocked = await invoke(execute, state, EXPAND, missing_aspect="   ")
    assert len(retriever.calls) == 1
    assert blocked["candidates"] == []
    assert "specific missing aspect" in blocked["errors"][0]
