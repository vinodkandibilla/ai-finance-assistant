from __future__ import annotations

import logging

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
Produce 1-3 useful dense search queries and exact sparse terms. Preserve tickers and named concepts."""


class FinanceQueryAnalyzer:
    def __init__(self, llm: object) -> None:
        self._structured_llm = llm.with_structured_output(QueryAnalysis)

    async def analyze(self, query: str) -> QueryAnalysis:
        try:
            result = await self._structured_llm.ainvoke(
                [SystemMessage(content=ANALYSIS_PROMPT), HumanMessage(content=query)]
            )
            analysis = result if isinstance(result, QueryAnalysis) else QueryAnalysis.model_validate(result)
            return self._normalize(analysis, query)
        except Exception:
            logger.exception("Finance query analysis failed; falling back to the original query.")
            fallback_intent: FinanceDomainIntent = "finance_qa"
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
        analysis.dense_sub_queries = [value.strip() for value in analysis.dense_sub_queries if value.strip()][:3]
        analysis.sparse_keywords = list(dict.fromkeys([
            *analysis.sparse_keywords,
            *deterministic_keywords(query),
        ]))
        return analysis


def deterministic_keywords(query: str) -> list[str]:
    terms = set(__import__("re").findall(r"\b[A-Z][A-Z0-9.-]{0,6}\b|\b\w{3,}\b", query))
    return sorted(terms, key=str.casefold)