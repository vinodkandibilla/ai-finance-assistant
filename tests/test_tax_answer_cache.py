"""Tax uses shared cache mechanics with narrow, separate eligibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

from finassist.agents.common.answer_cache import SemanticAnswerCache
from finassist.agents.tax_education.answer_cache import definition_scope
from finassist.workflow.builder import build_graph


@pytest.mark.parametrize(
    "query",
    [
        "What is the 2026 standard deduction?",
        "What is Form W-2 for a nonresident?",
        "What is a tax deduction in California?",
        "Compare deductions and credits",
        "Calculate tax on $100,000",
        "What is a tax deduction for married filing jointly?",
        "What is a tax credit? Give eligibility limits.",
    ],
)
def test_rule_specific_tax_requests_bypass(query):
    assert definition_scope(query) is None


async def test_tax_semantic_hit_skips_agent_and_returns_tax_intent():
    client = QdrantClient(":memory:")
    store = SemanticAnswerCache(
        SimpleNamespace(aembed_query=AsyncMock(return_value=[1.0, 0.0])),
        client,
        namespace="tax-2026-US-federal",
        scope=definition_scope,
        intent="tax",
        label="Tax",
        collection="tax_answer_cache",
        allowed_citations={"Wages and Withholding"},
    )
    compute = AsyncMock(
        return_value={
            "intent": "tax",
            "status": "answered",
            "answer": "Form W-2 reports wages and withholding.",
            "citations": ["Wages and Withholding"],
        }
    )
    graph = build_graph(
        SimpleNamespace(classify=AsyncMock(return_value="tax")),
        tax_agent=SimpleNamespace(graph=RunnableLambda(compute)),
        tax_answer_cache=store,
    )
    await graph.ainvoke({"query": "What is Form W-2?"})
    hit = await graph.ainvoke({"query": "Define W2"})
    assert hit["intent"] == "tax"
    assert compute.await_count == 1
    assert hit["user_query"] == "Define W2"
    assert client.collection_exists("tax_answer_cache")
    client.close()


async def test_namespace_prevents_reusing_another_tax_years_answer():
    client = QdrantClient(":memory:")
    embeddings = SimpleNamespace(aembed_query=AsyncMock(return_value=[1.0, 0.0]))
    arguments = dict(
        scope=definition_scope,
        intent="tax",
        collection="tax_answer_cache",
        allowed_citations={"Wages and Withholding"},
    )
    compute = AsyncMock(
        return_value={
            "status": "answered",
            "answer": "Wages and withholding.",
            "citations": ["Wages and Withholding"],
        }
    )
    old = SemanticAnswerCache(embeddings, client, namespace="tax-2025-US", **arguments)
    new = SemanticAnswerCache(embeddings, client, namespace="tax-2026-US", **arguments)
    await old.run("What is Form W-2?", compute)
    await new.run("What is Form W-2?", compute)
    assert compute.await_count == 2
    client.close()
