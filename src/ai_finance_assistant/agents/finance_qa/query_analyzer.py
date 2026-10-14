from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from ai_finance_assistant.agents.finance_qa.schemas import FinanceDomainIntent, QueryAnalysis
from ai_finance_assistant.prompts import ANALYSIS_PROMPT, TAX_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


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
