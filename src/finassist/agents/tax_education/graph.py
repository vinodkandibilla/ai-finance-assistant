"""Domain policy for the autonomous education agent."""

import re

from finassist.agents.common.rag_tools import tax_citations as _citations
from finassist.agents.common.runtime import AgentPolicy, AutonomousEducationAgent
from finassist.agents.tax_education.calculator_tools import (
    calculate_federal_ordinary_income_tax,
    process_tax_calculation_results,
)
from finassist.agents.tax_education.config import TaxAgentSettings
from finassist.agents.tax_education.evidence_recovery import (
    EXPAND,
    TARGET,
    create_tax_evidence_recovery,
)
from finassist.agents.tax_education.schemas import TaxAgentResult
from finassist.agents.tax_education.tools import create_tax_rag_query_tool
from finassist.prompts import TAX_EDUCATION_AGENT_PROMPT as AGENT_PROMPT
from finassist.rag.tax_education import extract_tax_metadata


class TaxEducationAgent(AutonomousEducationAgent):
    def __init__(self, *, llm, retriever, reranker, settings=None, checkpointer=None):
        settings = settings or TaxAgentSettings()
        baseline_tool = create_tax_rag_query_tool(
            retriever, reranker=reranker, rerank_top_n=settings.rerank_top_n
        )
        policy = AgentPolicy(
            intent="tax",
            recover_skipped_retrieval=True,
            label="Tax",
            name="tax_education_agent",
            prompt=AGENT_PROMPT,
            search_tool_name="tax_rag_query",
            clarification_tool_name="ask_tax_clarification",
            no_evidence_answer="I could not find supporting evidence in the tax corpus.",
            citations=_citations,
            search_arguments=_search_arguments,
            clarification_response=_clarification_response,
            process_tool_results=process_tax_calculation_results,
            recovery_tool_names=(EXPAND, TARGET),
            evidence_recovery=create_tax_evidence_recovery(
                retriever, reranker, settings, baseline_tool
            ),
        )
        super().__init__(
            llm=llm,
            retrieval=baseline_tool,
            settings=settings,
            policy=policy,
            result_schema=TaxAgentResult,
            checkpointer=checkpointer,
            additional_tools=[calculate_federal_ordinary_income_tax],
        )


def _search_arguments(query, original, dense_queries, sparse_keywords):
    return {
        "query": original,
        "search_query": query,
        "dense_queries": dense_queries,
        "sparse_keywords": sparse_keywords,
    }


def _clarification_response(question, original):
    metadata = extract_tax_metadata(original)
    if re.search(r"\bfiling\s+status\b|\bmarried\b|\bsingle\b", question, re.I):
        statuses = metadata.get("filing_status") or []
        if len(statuses) == 1:
            return (
                f"The user already supplied filing status: {statuses[0]}. "
                "Do not ask for it again. Use the original question and actual user replies. "
                "Search for the requested bracket if income is supplied; "
                "otherwise ask only for the missing income."
            )
    if re.search(r"\byear\b", question, re.I) and not re.search(
        r"\bincome\b|\bfiling status\b|\btopic\b|\bamount\b|\beligibility\b", question, re.I
    ):
        return (
            "Do not ask for year confirmation alone. Use the user's explicit year; "
            "if omitted, use the corpus scope: tax year 2026, filed in 2027. "
            "Continue with retrieval and state the applicable year in the answer."
        )
    return None
