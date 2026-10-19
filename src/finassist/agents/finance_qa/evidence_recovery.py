"""Bounded finance retrieval actions; the model chooses recovery order."""

from finassist.agents.common.evidence_recovery import (
    apply_evidence_budget,
    build_action_feedback,
    read_retrieval_history,
    remove_duplicate_passages,
)
from finassist.agents.common.rag_tools import finance_citations, rerank_evidence
from finassist.agents.common.runtime import user_replies
from finassist.agents.common.schemas import EvidenceResult

BASELINE = "finance_rag_query"
EXPAND = "finance_expand_evidence"
TARGET = "finance_targeted_search"
ACTIONS = {BASELINE, EXPAND, TARGET}


def create_evidence_recovery(retriever, metadata_extractor, reranker, settings):
    async def execute(action, query, state, dense=None, sparse=None, missing_aspect=""):
        # Reconstruct evidence usage and validate the requested action before retrieval.
        messages = state["messages"]
        calls, successful, seen, used_chars = read_retrieval_history(messages, ACTIONS)
        reason = _validate_action(
            action,
            query,
            dense,
            missing_aspect,
            messages,
            calls,
            successful,
            seen,
            used_chars,
            settings,
        )
        plan = _build_search_plan(action, query, dense, sparse, successful)
        original = _original_question_with_replies(messages)
        metadata = metadata_extractor.extract(original)
        result = EvidenceResult(metadata_filters=metadata)
        duplicates = 0
        if reason:
            result.errors.append(reason)
        else:
            try:
                # Retrieve at the selected depth while preserving original constraints.
                options, limit = _select_retrieval_profile(action, settings)
                documents, errors = await retriever.retrieve(
                    plan["dense"],
                    plan["sparse"],
                    metadata_filter=metadata_extractor.build_qdrant_filter(metadata),
                    **options,
                )
                novel, duplicates = remove_duplicate_passages(documents, seen)
                # Rank new evidence against the full question and enforce exposure budgets.
                ranked = EvidenceResult.model_validate(
                    await rerank_evidence(
                        query=original,
                        documents=novel,
                        metadata=metadata,
                        errors=errors,
                        reranker=reranker,
                        limit=limit,
                        citations=finance_citations,
                    )
                )
                used_chars = apply_evidence_budget(ranked, result, seen, used_chars, settings)
                result.errors = ranked.errors
                result.citations = finance_citations(result.candidates)
            except Exception as error:
                result.errors.append(f"Finance search failed: {type(error).__name__}")
        payload = build_action_feedback(
            result,
            action,
            missing_aspect,
            duplicates,
            calls,
            seen,
            used_chars,
            settings,
        )
        # Retain error-free search terms for expansion, including searches with no results.
        if not reason and not result.errors:
            payload["search_plan"] = plan
        return payload, payload

    return execute


def _validate_action(
    action, query, dense, missing_aspect, messages, calls, successful, seen, used_chars, settings
):
    reason = None
    if len(calls) > settings.agent_max_searches:
        reason = "Finance search budget exhausted."
    elif action == BASELINE and sum(c["name"] == BASELINE for c in calls) > 1:
        reason = "Baseline acquisition already attempted; choose expansion or targeted search."
    elif action != BASELINE and not successful:
        reason = "Acquire baseline evidence before recovery."
    elif (
        action == EXPAND
        and sum(c["name"] == EXPAND for c in calls) > settings.max_expansion_actions
    ):
        reason = "Expansion action limit reached."
    elif (
        action == TARGET and sum(c["name"] == TARGET for c in calls) > settings.max_targeted_actions
    ):
        reason = "Targeted-search action limit reached."
    elif action != EXPAND and not query.strip():
        reason = "Supply a nonempty search query."
    elif action == TARGET and not missing_aspect.strip():
        reason = "Identify the missing aspect before targeted search."
    elif len(dense or []) > settings.max_dense_queries:
        reason = "Semantic query limit exceeded."
    elif (
        len(seen) >= settings.max_evidence_passages
        or used_chars >= settings.max_evidence_characters
    ):
        reason = "Evidence budget exhausted."

    return reason


def _build_search_plan(action, query, dense, sparse, successful):
    plan = {
        "query": query,
        "dense": dense or [query],
        "sparse": sparse if sparse is not None else [query],
    }
    if action == EXPAND and successful:
        plan = successful[-1]
    return plan


def _original_question_with_replies(messages):
    original = str(messages[0].content)
    for reply in user_replies(
        messages, "ask_finance_clarification", ("ask_interest_calculation_clarification",)
    ):
        original += "\nAdditional details: " + reply
    return original


def _select_retrieval_profile(action, settings):
    options = {}
    limit = settings.rerank_top_n
    if action == EXPAND:
        options["profile"] = {
            "dense_limit": settings.expansion_dense_top_k,
            "sparse_limit": settings.expansion_sparse_top_k,
            "candidate_limit": settings.expansion_rrf_candidates,
        }
        limit = settings.expansion_rerank_top_n
    return options, limit
