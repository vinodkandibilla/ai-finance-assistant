"""Narrow standalone-greeting recognition shared by all domain workspaces."""

import re

WELCOME = (
    "Hi! I’m FinPilot. I can help with finance concepts, portfolios, market data, "
    "savings goals, financial news, and US taxes. What would you like to explore?"
)


def is_standalone_greeting(query: str) -> bool:
    # Market chat wraps questions with exchange/format instructions.
    if query.startswith("Market scope:") and "\nQuestion: " in query:
        query = query.split("\nQuestion: ", 1)[1]
    return bool(
        re.fullmatch(
            r"\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))"
            r"(?:\s*[,!]?[ ]*(?:finpilot|there))?[.!?\s]*",
            query,
            re.IGNORECASE,
        )
    )
