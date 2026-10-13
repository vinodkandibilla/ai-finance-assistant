"""Compound-interest calculator using shared Decimal arithmetic."""

from ai_finance_assistant.agents.finance_qa.interest_calculation import (
    CompoundInterestInputs as CompoundInterestInputs,
)
from ai_finance_assistant.agents.finance_qa.interest_calculation import (
    calculate_interest,
)


def calculate(inputs: dict) -> dict:
    return calculate_interest(inputs, kind="compound")
