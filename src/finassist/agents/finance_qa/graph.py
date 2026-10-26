"""Domain policy for the autonomous education agent."""

import re

from finassist.agents.common.rag_tools import finance_citations as _citations
from finassist.agents.common.runtime import AgentPolicy, AutonomousEducationAgent
from finassist.agents.finance_qa.calculator_tools import (
    CLARIFICATION,
    create_calculation_clarification_tool,
    create_calculation_result_processor,
    create_calculation_tools,
)
from finassist.agents.finance_qa.evidence_recovery import (
    EXPAND,
    TARGET,
    create_evidence_recovery,
)
from finassist.agents.finance_qa.schemas import FinanceAgentResult, FinanceDocument
from finassist.agents.finance_qa.tools import create_finance_rag_query_tool
from finassist.prompts import FINANCE_QA_AGENT_PROMPT as AGENT_PROMPT


class FinanceQaAgent(AutonomousEducationAgent):
    def __init__(
        self,
        *,
        llm,
        retriever,
        metadata_extractor,
        classifier,
        reranker,
        settings,
        checkpointer=None,
    ):
        policy = AgentPolicy(
            intent="finance_qa",
            completion_tool_name="submit_finance_answer",
            recover_skipped_retrieval=True,
            label="Finance",
            name="finance_education_agent",
            prompt=AGENT_PROMPT,
            search_tool_name="finance_rag_query",
            clarification_tool_name="ask_finance_clarification",
            no_evidence_answer="I could not find supporting evidence in the finance handbook.",
            citations=_citations,
            search_arguments=_search_arguments,
            process_tool_results=create_calculation_result_processor(),
            clarification_response=_calculation_clarification,
            additional_reply_tools=(CLARIFICATION,),
            recovery_tool_names=(EXPAND, TARGET),
            evidence_recovery=create_evidence_recovery(
                retriever, metadata_extractor, reranker, settings
            ),
        )
        super().__init__(
            llm=llm,
            retrieval=create_finance_rag_query_tool(
                retriever, metadata_extractor, reranker=reranker, rerank_top_n=settings.rerank_top_n
            ),
            settings=settings,
            policy=policy,
            result_schema=FinanceAgentResult,
            checkpointer=checkpointer,
            additional_tools=[
                *create_calculation_tools(),
                *create_calculation_tools(kind="simple"),
                create_calculation_clarification_tool(),
            ],
        )


def _search_arguments(query, original, dense_queries, sparse_keywords):
    return {"query": query, "dense_queries": dense_queries, "sparse_keywords": sparse_keywords}


def _format_context(documents: list[FinanceDocument]) -> str:
    return "\n\n".join(
        f"[Source: Financial Education Handbook; "
        f"page {doc.metadata.get('page_number', 'unknown')}]\n{doc.content}"
        for doc in documents
    )


def _calculation_clarification(question, original):
    if re.search(r"(?:compound|simple)[- ]interest", original, re.I) and re.search(
        r"calculate|how much|earn|\$|principal|balance", original, re.I
    ):
        return (
            f"For calculation details, use {CLARIFICATION}. "
            "It asks all unresolved inputs together and enforces the three-message limit. "
            "Do not ask separate calculation clarification questions."
        )
    return None
