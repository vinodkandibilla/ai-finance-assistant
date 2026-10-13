"""Simple-interest calculator using shared Decimal arithmetic."""

from ai_finance_assistant.agents.finance_qa.interest_calculation import (
    SimpleInterestInputs as SimpleInterestInputs,
)
from ai_finance_assistant.agents.finance_qa.interest_calculation import (
    calculate_interest,
)


def calculate(inputs: dict) -> dict:
    return calculate_interest(inputs, kind="simple")
