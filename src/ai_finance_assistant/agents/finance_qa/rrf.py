from __future__ import annotations

from collections import defaultdict

from ai_finance_assistant.agents.finance_qa.schemas import FinanceDocument


def reciprocal_rank_fusion(
    ranked_lists: list[list[FinanceDocument]], *, k: int = 60, limit: int = 20
) -> list[FinanceDocument]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, FinanceDocument] = {}
    provenance: dict[str, list[str]] = defaultdict(list)

    for list_index, ranked_list in enumerate(ranked_lists):
        for rank, document in enumerate(ranked_list, start=1):
            scores[document.doc_id] += 1.0 / (k + rank)
            documents.setdefault(document.doc_id, document)
            provenance[document.doc_id].append(f"ranked_list_{list_index + 1}:rank_{rank}")

    fused = [
        document.model_copy(
            update={
                "score": scores[doc_id],
                "rank": 0,
                "provenance": provenance[doc_id],
            }
        )
        for doc_id, document in documents.items()
    ]
    fused.sort(key=lambda document: (-document.score, document.doc_id))
    return [document.model_copy(update={"rank": rank}) for rank, document in enumerate(fused[:limit], 1)]