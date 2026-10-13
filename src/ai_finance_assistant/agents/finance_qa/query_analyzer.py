from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from ai_finance_assistant.agents.finance_qa.schemas import FinanceDomainIntent, QueryAnalysis

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """Analyze a query for an investment-education search system.
In scope: general investing concepts, securities, funds, benchmarks, options mechanics,
and investment risk education. Out of scope: retirement accounts, taxes, immigration,
live prices, breaking news, personalized portfolios/planning, trades, and account actions.
Return detected intents for every requested domain. Use finance_qa for in-scope questions;
use tax, market, news, portfolio, goals, or other for excluded requests. Do not answer.
Clarify only if essential comparands or the subject are missing. 'What is Delta?' is searchable.
Produce 1-3 useful dense search queries and exact sparse terms.
Preserve tickers and named concepts."""

TAX_ANALYSIS_PROMPT = """Analyze a query for an IRS tax-education search system.
Return detected intent tax. Break the question into 1-3 precise dense retrieval queries and
exact sparse terms that preserve filing status, tax year, form numbers, amounts, and named
tax concepts. Focus on the rule the user asks about, not merely incidental terms. Do not
answer the question. For a general definition such as 'What is a standard deduction?',
keep the conceptual question; do not invent a tax year, ask for filing status, or search for
amounts unless the user requests an amount or eligibility determination. Clarify only when
the question is very vague and its tax subject or essential inputs cannot be identified.
The corpus covers tax year 2026, generally filed in 2027; an omitted year alone never
requires clarification. Preserve explicit years and do not invent a tax year in search terms.
For bracket questions, treat unqualified income as taxable ordinary income and normalize
130K to $130,000. Income plus filing status is enough to search without clarification.
A personal bracket lookup such as 'What will be my tax bracket' requires both an income
amount and filing status. If either is missing, set needs_clarification=true and ask only
for the missing inputs before retrieval. A general explanation of tax brackets does not
require these personal inputs.
Do not reinterpret explicit gross income, salary, wages, or take-home pay as taxable income."""


class FinanceQueryAnalyzer:
    def __init__(self, llm: object, *, domain: Literal["finance_qa", "tax"] = "finance_qa") -> None:
        self._structured_llm = llm.with_structured_output(QueryAnalysis)
        self._domain = domain
        self._prompt = TAX_ANALYSIS_PROMPT if domain == "tax" else ANALYSIS_PROMPT

    async def analyze(self, query: str) -> QueryAnalysis:
        try:
            result = await self._structured_llm.ainvoke(
                [SystemMessage(content=self._prompt), HumanMessage(content=query)]
            )
            analysis = (
                result
                if isinstance(result, QueryAnalysis)
                else QueryAnalysis.model_validate(result)
            )
            return self._normalize(analysis, query)
        except Exception:
            logger.exception("Finance query analysis failed; falling back to the original query.")
            fallback_intent: FinanceDomainIntent = self._domain
            return QueryAnalysis(
                needs_clarification=False,
                detected_intents=[fallback_intent],
                dense_sub_queries=[query],
                sparse_keywords=deterministic_keywords(query),
            )

    @staticmethod
    def _normalize(analysis: QueryAnalysis, query: str) -> QueryAnalysis:
        if not analysis.dense_sub_queries:
            analysis.dense_sub_queries = [query]
        analysis.dense_sub_queries = [
            value.strip() for value in analysis.dense_sub_queries if value.strip()
        ][:3]
        analysis.sparse_keywords = list(
            dict.fromkeys(
                [
                    *analysis.sparse_keywords,
                    *deterministic_keywords(query),
                ]
            )
        )
        return analysis


def deterministic_keywords(query: str) -> list[str]:
    terms = set(__import__("re").findall(r"\b[A-Z][A-Z0-9.-]{0,6}\b|\b\w{3,}\b", query))
    return sorted(terms, key=str.casefold)
