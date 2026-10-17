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


ExecutedHandler = Literal[
    "finance_qa_node",
    "tax_education_node",
    "market_analysis_node",
    "portfolio_analysis_node",
    "goals_node",
    "news_synthesizer_node",
    "clarify_node",
    "unsupported_node",
]


class FinanceAssistantState(TypedDict, total=False):
    portfolio_user_context: dict[str, str] | None
    customer_id: str
    manager_id: str
    user_role: str
    query: str
    intent: FinanceIntent
    classified_intent: FinanceIntent
    executed_handler: ExecutedHandler
    standalone_greeting: bool
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
    clarification_question: str
    calculations: list[dict]
    calculation: dict
    final_result: dict[str, object]
    citations: list[str]
    status: str
    thread_id: str
