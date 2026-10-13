"""Deterministic postprocessing inside domain RAG tools."""

from ai_finance_assistant.agents.common.schemas import EvidenceResult
from ai_finance_assistant.rag.models import RetrievedDocument


def finance_citations(documents: list[RetrievedDocument]) -> list[str]:
    return list(
        dict.fromkeys(
            f"Financial Education Handbook, page {doc.metadata.get('page_number', 'unknown')}"
            for doc in documents
        )
    )


def tax_citations(documents: list[RetrievedDocument]) -> list[str]:
    return list(
        dict.fromkeys(str(doc.metadata.get("title") or "Tax education corpus") for doc in documents)
    )


async def rerank_evidence(*, query, documents, metadata, errors, reranker, limit, citations):
    """Rank retrieved evidence and produce the common tool result in one invocation."""
    if documents:
        documents = await reranker.rerank(query, documents, limit=limit)
    return EvidenceResult(
        candidates=documents,
        metadata_filters=metadata,
        errors=errors,
        citations=citations(documents),
    ).model_dump()
