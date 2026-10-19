"""Shared bookkeeping and budget helpers for domain evidence recovery."""

from langchain_core.messages import ToolMessage

from finassist.agents.common.schemas import EvidenceResult


def read_retrieval_history(messages, actions):
    calls = [c for m in messages for c in getattr(m, "tool_calls", []) if c["name"] in actions]
    previous = [
        m.artifact or {} for m in messages if isinstance(m, ToolMessage) and m.name in actions
    ]
    plans = [item["search_plan"] for item in previous if item.get("search_plan")]
    seen = {}
    for item in previous:
        for raw in item.get("candidates", []):
            doc = EvidenceResult(candidates=[raw]).candidates[0]
            seen[(doc.source, doc.doc_id, doc.content)] = doc
    return calls, plans, seen, sum(len(doc.content) for doc in seen.values())


def remove_duplicate_passages(documents, seen):
    novel, duplicates, keys = [], 0, set(seen)
    for doc in documents:
        key = (doc.source, doc.doc_id, doc.content)
        if key in keys:
            duplicates += 1
        else:
            keys.add(key)
            novel.append(doc)
    return novel, duplicates


def apply_evidence_budget(ranked, result, seen, used_chars, settings):
    for doc in ranked.candidates:
        if len(seen) + len(result.candidates) >= settings.max_evidence_passages:
            break
        if used_chars + len(doc.content) <= settings.max_evidence_characters:
            result.candidates.append(doc)
            used_chars += len(doc.content)
    return used_chars


def build_action_feedback(
    result, action, missing_aspect, duplicates, calls, seen, used_chars, settings
):
    payload = result.model_dump()
    payload.update(
        action=action,
        profile="expansion" if action.endswith("expand_evidence") else "baseline",
        missing_aspect=missing_aspect,
        duplicates_skipped=duplicates,
        new_passages=len(result.candidates),
        remaining_searches=max(0, settings.agent_max_searches - len(calls)),
        remaining_expansions=max(
            0,
            settings.max_expansion_actions
            - sum(c["name"].endswith("expand_evidence") for c in calls),
        ),
        remaining_targeted_searches=max(
            0,
            settings.max_targeted_actions
            - sum(c["name"].endswith("targeted_search") for c in calls),
        ),
        remaining_passages=max(
            0, settings.max_evidence_passages - len(seen) - len(result.candidates)
        ),
        remaining_characters=max(0, settings.max_evidence_characters - used_chars),
    )
    return payload
