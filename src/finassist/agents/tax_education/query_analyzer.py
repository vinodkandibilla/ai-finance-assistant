import re

from finassist.agents.finance_qa.query_analyzer import FinanceQueryAnalyzer
from finassist.agents.finance_qa.schemas import QueryAnalysis


class TaxQueryAnalyzer(FinanceQueryAnalyzer):
    """Tax entry point using the existing structured-analysis implementation.

    This transitional adapter preserves normalization and fallback behavior while
    giving tax orchestration an explicit dependency on tax-domain analysis.
    """

    def __init__(self, llm: object) -> None:
        super().__init__(llm, domain="tax")

    async def analyze(self, query: str) -> QueryAnalysis:
        question = _bracket_clarification(query)
        if question:
            return QueryAnalysis(
                needs_clarification=True,
                clarification_question=question,
                detected_intents=["tax"],
                dense_sub_queries=[query],
            )
        analysis = await super().analyze(query)
        # Preserve the user's question as the primary search. Decomposition must
        # not turn a concept definition into a year-specific amount lookup.
        years = set(re.findall(r"\b\d{4}\b", query))
        supported_queries = [
            value
            for value in analysis.dense_sub_queries
            if set(re.findall(r"\b\d{4}\b", value)).issubset(years)
        ]
        analysis.dense_sub_queries = list(dict.fromkeys([query, *supported_queries]))[:3]
        return analysis


def _bracket_clarification(query: str) -> str | None:
    """Require the two inputs for a personal bracket lookup before retrieval."""
    if not re.search(r"\btax\s+brackets?\b|\bmarginal\s+tax\s+rate\b", query, re.I):
        return None
    if not re.search(r"\b(?:my|i|me)\b", query, re.I):
        return None
    # A year is not an income amount. Remove years before checking numeric inputs.
    without_years = re.sub(r"\b20\d{2}\b", "", query)
    has_income = bool(
        re.search(
            r"\$\s*\d|\b\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|million)\b"
            r"|\b(?:income|earn|earning|make|salary|wages)\s*(?:of|is|:|=)?\s*\$?\s*\d",
            without_years,
            re.I,
        )
    )
    has_status = bool(
        re.search(
            r"\b(?:single|married(?:\s+filing)?\s+(?:jointly|separately)|mfj|mfs|"
            r"head\s+of\s+household|hoh|qualifying\s+surviving\s+spouse|qss)\b",
            query,
            re.I,
        )
    )
    if not has_income and not has_status:
        return "What is your taxable income and filing status (Single, MFJ, MFS, HOH, or QSS)?"
    if not has_income:
        return "What is your taxable income?"
    if not has_status:
        return "What is your filing status (Single, MFJ, MFS, HOH, or QSS)?"
    return None
