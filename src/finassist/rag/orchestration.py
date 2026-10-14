"""Shared bounded-retrieval and evidence-grounded generation orchestration."""

from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from finassist.rag.evaluation import ContextEvaluation


def should_replace_best(previous: ContextEvaluation | None, current: ContextEvaluation) -> bool:
    return previous is None or (current.sufficient and not previous.sufficient)


def select_context(
    documents: list[dict], current: ContextEvaluation, previous: dict | None
) -> dict:
    update = {"evaluation": current.model_dump()}
    prior = ContextEvaluation.model_validate(previous) if previous else None
    if should_replace_best(prior, current):
        update.update(best_context=documents, best_evaluation=current.model_dump())
    return update


def route_evaluation(
    evaluation: dict, retry_count: int, max_retries: int
) -> Literal["generate", "reformulate"]:
    if ContextEvaluation.model_validate(evaluation).sufficient or retry_count >= max_retries:
        return "generate"
    return "reformulate"


def reformulate_query(
    query: str,
    evaluation: dict,
    keywords: list[str],
    retry_count: int,
    *,
    update_search_query: bool = False,
) -> dict:
    assessment = ContextEvaluation.model_validate(evaluation)
    queries = assessment.suggested_queries or [
        f"{query} {' '.join(assessment.missing_concepts)}".strip()
    ]
    update = {
        "retry_count": retry_count + 1,
        "dense_sub_queries": queries[:3],
        "sparse_keywords": list(dict.fromkeys([*keywords, *assessment.missing_concepts])),
    }
    if update_search_query:
        terms = assessment.suggested_queries or assessment.missing_concepts
        update["search_query"] = query + "\nSearch focus: " + " ".join(terms[:3])
    return update


@dataclass(frozen=True)
class GenerationPolicy:
    system_prompt: str
    no_evidence_answer: str
    insufficient_answer: str
    retrieval_error_answer: str
    generation_error_answer: str
    errors_override_insufficient: bool = False


async def generate_answer(
    *,
    llm: object,
    query: str,
    has_documents: bool,
    evaluation: dict | None,
    context: str,
    citations: list[str],
    errors: list[str],
    policy: GenerationPolicy,
) -> dict:
    sufficient = bool(evaluation and ContextEvaluation.model_validate(evaluation).sufficient)
    if not has_documents or not sufficient:
        failed = bool(errors) and (not has_documents or policy.errors_override_insufficient)
        answer = (
            policy.retrieval_error_answer
            if failed
            else policy.insufficient_answer
            if has_documents
            else policy.no_evidence_answer
        )
        return {
            "answer": answer,
            "status": "error" if failed else "insufficient",
            "citations": citations,
            "errors": errors if failed else [],
        }
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=policy.system_prompt),
                HumanMessage(content=f"Question: {query}\n\nEvidence:\n{context}"),
            ]
        )
        answer = response if isinstance(response, str) else getattr(response, "content", "")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty generated answer")
        return {
            "answer": answer.strip(),
            "status": "answered",
            "citations": citations,
            "errors": [],
        }
    except Exception as error:
        return {
            "answer": policy.generation_error_answer,
            "status": "error",
            "citations": citations,
            "errors": [*errors, type(error).__name__],
        }
