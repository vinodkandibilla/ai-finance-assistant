"""Deterministic postprocessing inside domain RAG tools."""

from langsmith import trace

from finassist.agents.common.schemas import EvidenceResult
from finassist.rag.models import RetrievedDocument


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
    if reranker is None:
        with trace(
            "Select top RRF evidence",
            run_type="retriever",
            inputs={"candidate_count": len(documents), "limit": limit},
            metadata={"reranker_enabled": False},
        ) as span:
            documents = documents[:limit]
            span.add_outputs({"selected_count": len(documents)})
    elif documents:
        documents = await reranker.rerank(query, documents, limit=limit)
    return EvidenceResult(
        candidates=documents,
        metadata_filters=metadata,
        errors=errors,
        citations=citations(documents),
    ).model_dump()
