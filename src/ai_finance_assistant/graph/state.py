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
