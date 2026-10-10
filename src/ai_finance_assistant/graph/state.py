from typing import Literal, TypedDict

FinanceIntent = Literal[
    "finance_qa",
    "portfolio",
    "market",
    "goals",
    "news",
    "tax",
    "clarify",
    "other",
]


class FinanceAssistantState(TypedDict, total=False):
    query: str
    intent: FinanceIntent
    answer: str
    user_query: str
    search_query: str
    analysis: dict[str, object]
    metadata_filters: dict[str, str]
    dense_sub_queries: list[str]
    sparse_keywords: list[str]
    ranked_lists: list[list[dict[str, object]]]
    sparse_ranked_list: list[dict[str, object]]
    candidates: list[dict[str, object]]
    reranked_documents: list[dict[str, object]]
    evaluation: dict[str, object]
    best_context: list[dict[str, object]]
    best_evaluation: dict[str, object]
    retry_count: int
    errors: list[str]
    handoff_intents: list[str]
    clarification_question: str
    final_result: dict[str, object]
    citations: list[str]
    status: str
    thread_id: str
