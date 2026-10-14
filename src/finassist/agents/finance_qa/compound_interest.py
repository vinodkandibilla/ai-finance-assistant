"""Compound-interest calculator using shared Decimal arithmetic."""

from finassist.agents.finance_qa.interest_calculation import (
    CompoundInterestInputs as CompoundInterestInputs,
)
from finassist.agents.finance_qa.interest_calculation import (
    calculate_interest,
)


def calculate(inputs: dict) -> dict:
    return calculate_interest(inputs, kind="compound")
