"""Retrieval stage spans stay under the request, including worker-thread scoring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from langsmith import Client, tracing_context
from langsmith.run_trees import RunTree

from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from finassist.agents.finance_qa.reranker import FinanceCrossEncoderReranker
from finassist.agents.finance_qa.retrieval import FinanceHybridRetriever
from finassist.agents.finance_qa.tools import create_finance_rag_query_tool


async def test_retrieval_stage_spans_preserve_parentage_and_compact_payloads():
    client = Mock(spec=Client)
    parent = RunTree(name="request", run_type="chain", client=client)
    qdrant = Mock()
    qdrant.collection_exists.return_value = True
    qdrant.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="chunk-1",
                score=0.9,
                payload={"page_content": "PRIVATE_DOCUMENT_TEXT", "metadata": {"page_number": 9}},
            )
        ]
    )
    retriever = FinanceHybridRetriever(
        [],
        qdrant_client=qdrant,
        embeddings=SimpleNamespace(aembed_query=AsyncMock(return_value=[0.1, 0.2])),
        sparse_embeddings=SimpleNamespace(
            embed_query=lambda query: SimpleNamespace(
                indices=[1],
                values=[0.5],
            )
        ),
        settings=FinanceAgentSettings(max_parallel_searches=1),
    )
    reranker = FinanceCrossEncoderReranker(
        "test-model",
        model=SimpleNamespace(predict=lambda pairs, **kwargs: [0.8]),
    )
    tool = create_finance_rag_query_tool(
        retriever,
        FinanceMetadataExtractor([]),
        reranker=reranker,
    )
    with tracing_context(parent=parent, enabled=True, client=client):
        result = await tool.ainvoke({"query": "PRIVATE_QUERY_TEXT"})

    def descendants(run):
        return [child for direct in run.child_runs for child in [direct, *descendants(direct)]]

    runs = descendants(parent)
    by_name = {run.name: run for run in runs}
    expected = {
        "Extract finance metadata",
        "Retrieve finance candidates",
        "Wait for retrieval slot",
        "Embed dense queries",
        "Encode sparse query",
        "Qdrant hybrid search and RRF",
        "Convert Qdrant results",
        "Rerank and prepare finance evidence",
        "Rerank evidence",
        "Wait for reranker lock",
        "Get or load reranker model",
        "Score evidence pairs",
    }
    assert expected <= by_name.keys()
    for name in expected:
        run = by_name[name]
        assert run.trace_id == parent.trace_id
        assert run.end_time is not None
        assert "PRIVATE" not in str(run.inputs) + str(run.outputs)
    assert by_name["Score evidence pairs"].parent_run_id == by_name["Rerank evidence"].id
    assert by_name["Qdrant hybrid search and RRF"].outputs == {"result_count": 1}
    assert result["candidates"][0]["content"] == "PRIVATE_DOCUMENT_TEXT"
    assert retriever._semaphore._value == 1
