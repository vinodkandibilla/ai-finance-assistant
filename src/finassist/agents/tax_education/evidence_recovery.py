"""Bounded tax evidence recovery actions."""

from finassist.agents.common.evidence_recovery import (
    apply_evidence_budget,
    build_action_feedback,
    read_retrieval_history,
    remove_duplicate_passages,
)
from finassist.agents.common.rag_tools import rerank_evidence, tax_citations
from finassist.agents.common.runtime import user_replies
from finassist.agents.common.schemas import EvidenceResult
from finassist.rag.tax_education import build_tax_metadata_filter, extract_tax_metadata

BASELINE = "tax_rag_query"
EXPAND = "tax_expand_evidence"
TARGET = "tax_targeted_search"
ACTIONS = {BASELINE, EXPAND, TARGET}


def create_tax_evidence_recovery(retriever, reranker, settings):
    async def execute(action, query, state, dense=None, sparse=None, missing_aspect=""):
        messages = state["messages"]
        calls, plans, seen, used_chars = read_retrieval_history(messages, ACTIONS)
        reason = _validate(action, query, missing_aspect, calls, plans, seen, used_chars, settings)
        plan = (
            plans[-1]
            if action == EXPAND and plans
            else {
                "query": query,
                "dense": dense or [query],
                "sparse": sparse if sparse is not None else [query],
            }
        )
        original = str(messages[0].content)
        for reply in user_replies(messages, "ask_tax_clarification"):
            original += "\nAdditional details: " + reply
        metadata = extract_tax_metadata(original)
        result = EvidenceResult(metadata_filters=metadata)
        duplicates = 0
        if reason:
            result.errors.append(reason)
        else:
            try:
                documents = await retriever.search(
                    plan["query"],
                    limit=settings.expansion_search_limit if action == EXPAND else 20,
                    metadata_filter=build_tax_metadata_filter(metadata),
                    dense_queries=plan["dense"],
                    sparse_query=" ".join(plan["sparse"]),
                )
                documents = [
                    d
                    for d in documents
                    if all(
                        str(d.metadata.get(f)) == str(metadata[f])
                        for f in ("tax_year", "filing_year")
                        if f in metadata
                    )
                ]
                novel, duplicates = remove_duplicate_passages(documents, seen)
                ranked = EvidenceResult.model_validate(
                    await rerank_evidence(
                        query=original,
                        documents=novel,
                        metadata=metadata,
                        errors=[],
                        reranker=reranker,
                        limit=settings.expansion_rerank_top_n
                        if action == EXPAND
                        else settings.rerank_top_n,
                        citations=tax_citations,
                    )
                )
                used_chars = apply_evidence_budget(ranked, result, seen, used_chars, settings)
                result.errors = ranked.errors
                result.citations = tax_citations(result.candidates)
            except Exception as error:
                result.errors.append(f"Tax search failed: {type(error).__name__}")
        payload = build_action_feedback(
            result, action, missing_aspect, duplicates, calls, seen, used_chars, settings
        )
        if not reason and not result.errors:
            payload["search_plan"] = plan
        return payload, payload

    return execute


def _validate(action, query, missing_aspect, calls, plans, seen, used_chars, settings):
    if len(calls) > settings.agent_max_searches:
        return "Tax search budget exhausted."
    if action == BASELINE and sum(c["name"] == BASELINE for c in calls) > 1:
        return "Baseline acquisition already attempted; choose expansion or targeted search."
    if action != BASELINE and not plans:
        return "Acquire baseline evidence before recovery."
    if (
        action == EXPAND
        and sum(c["name"] == EXPAND for c in calls) > settings.max_expansion_actions
    ):
        return "Expansion action limit reached."
    if action == TARGET and sum(c["name"] == TARGET for c in calls) > settings.max_targeted_actions:
        return "Targeted-search action limit reached."
    if action != EXPAND and not query.strip():
        return "Supply a nonempty search query."
    if action in {TARGET, EXPAND} and not missing_aspect.strip():
        return "Identify the specific missing aspect before recovery."
    if (
        len(seen) >= settings.max_evidence_passages
        or used_chars >= settings.max_evidence_characters
    ):
        return "Evidence budget exhausted."
    return None
