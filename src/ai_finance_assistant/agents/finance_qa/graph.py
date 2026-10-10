from __future__ import annotations

import logging
import re
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ai_finance_assistant.agents.finance_qa.evaluator import FinanceContextEvaluator
from ai_finance_assistant.agents.finance_qa.metadata_extractor import FinanceMetadataExtractor
from ai_finance_assistant.agents.finance_qa.rrf import reciprocal_rank_fusion
from ai_finance_assistant.agents.finance_qa.reranker import FinanceCrossEncoderReranker
from ai_finance_assistant.agents.finance_qa.retrieval import FinanceHybridRetriever
from ai_finance_assistant.agents.finance_qa.schemas import (
    ContextEvaluation,
    FinanceAgentResult,
    FinanceDocument,
    QueryAnalysis,
)
from ai_finance_assistant.agents.finance_qa.query_analyzer import FinanceQueryAnalyzer
from ai_finance_assistant.graph.classifier import IntentClassifier

logger = logging.getLogger(__name__)

SCOPE_EXCLUSIONS = (
    re.compile(r"\b(?:roth|traditional|sep|simple)\s+ira\b|\b401\s*\(?k\)?\b|\bretirement\b", re.I),
    re.compile(r"\b(?:tax|taxation|irs|deduction|filing status|tax credit)\b", re.I),
    re.compile(r"\b(?:immigration|visa|resident alien|nonresident alien)\b", re.I),
    re.compile(r"\b(?:today|right now|current|latest)\b.{0,32}\b(?:stock price|quote|market price)\b", re.I),
    re.compile(r"\b(?:buy|sell|trade|execute|place)\b.{0,20}\b(?:shares|stock|option|trade)\b", re.I),
)


class FinanceAgentState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    status: str
    citations: list[str]
    user_query: str
    search_query: str
    analysis: dict[str, object]
    metadata_filters: dict[str, str]
    dense_sub_queries: list[str]
    sparse_keywords: list[str]
    ranked_lists: list[list[dict[str, object]]]
    sparse_ranked_list: list[dict[str, object]]
    candidates: list[dict[str, object]]
    reranked_documents: list[dict[str, object]]
    evaluation: dict[str, object]
    best_context: list[dict[str, object]]
    best_evaluation: dict[str, object]
    retry_count: int
    errors: list[str]
    handoff_intents: list[str]
    clarification_question: str
    final_result: dict[str, object]


class FinanceQaAgent:
    def __init__(
        self,
        *,
        llm: object,
        retriever: FinanceHybridRetriever,
        metadata_extractor: FinanceMetadataExtractor,
        classifier: IntentClassifier,
        reranker: FinanceCrossEncoderReranker,
        settings: object,
        checkpointer: object | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._metadata_extractor = metadata_extractor
        self._classifier = classifier
        self._reranker = reranker
        self._settings = settings
        self._checkpointer = checkpointer
        self._analyzer = FinanceQueryAnalyzer(llm)
        self._evaluator = FinanceContextEvaluator(llm)
        self.graph = self._build_graph()

    async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        configurable = dict(config.get("configurable", {}))
        configurable.setdefault("thread_id", "finance-agent-default")
        config = {**config, "configurable": configurable}
        result = await self.graph.ainvoke(state, config=config)
        return result

    def _build_graph(self):
        async def analyze_node(state: FinanceAgentState) -> dict[str, object]:
            original_query = str(state.get("user_query") or state.get("query") or "")
            query = str(state.get("search_query") or original_query)
            forced = _deterministic_handoff(query)
            if forced:
                analysis = QueryAnalysis(
                    needs_clarification=False,
                    detected_intents=forced,
                    dense_sub_queries=[query],
                )
            else:
                analysis = await self._analyzer.analyze(query)
            intents = analysis.detected_intents or ["finance_qa"]
            handoff = [intent for intent in intents if intent != "finance_qa"]
            if handoff:
                return {
                    "user_query": original_query,
                    "analysis": analysis.model_dump(),
                    "handoff_intents": [str(intent) for intent in intents],
                }
            return {
                "user_query": original_query,
                "analysis": analysis.model_dump(),
                "dense_sub_queries": analysis.dense_sub_queries or [query],
                "sparse_keywords": analysis.sparse_keywords,
                "clarification_question": analysis.clarification_question or "",
            }

        async def clarify_node(state: FinanceAgentState) -> dict[str, object]:
            question = str(state.get("clarification_question") or "Could you clarify the financial concept or comparison?")
            user_reply = interrupt({"kind": "clarification", "question": question})
            resumed_intent = await self._classifier.classify(str(user_reply))
            if resumed_intent != "finance_qa":
                return {
                    "intent": resumed_intent,
                    "handoff_intents": [resumed_intent],
                    "search_query": str(user_reply),
                }
            original_query = str(state.get("user_query") or state.get("query") or "")
            return {
                "search_query": f"{original_query}\nAdditional details: {user_reply}",
                "retry_count": 0,
                "errors": [],
            }

        async def retrieve_node(state: FinanceAgentState) -> dict[str, object]:
            analysis = QueryAnalysis.model_validate(state.get("analysis") or {})
            search_query = str(state.get("search_query") or state["user_query"])
            if state.get("retry_count", 0):
                dense_queries = list(state.get("dense_sub_queries") or analysis.dense_sub_queries or [search_query])
                keywords = analysis.sparse_keywords
            else:
                dense_queries = list(state.get("dense_sub_queries") or [search_query])
                keywords = list(state.get("sparse_keywords") or [])
            metadata = self._metadata_extractor.extract(search_query)
            metadata_filter = self._metadata_extractor.build_qdrant_filter(metadata)
            ranked_lists, sparse_list, errors = await self._retriever.retrieve(
                dense_queries,
                keywords,
                metadata_filter=metadata_filter,
            )
            if errors and not ranked_lists and not sparse_list:
                errors.append("both dense and sparse retrieval returned no usable channel")
            return {
                "metadata_filters": metadata,
                "ranked_lists": [[doc.model_dump() for doc in values] for values in ranked_lists],
                "sparse_ranked_list": [doc.model_dump() for doc in sparse_list],
                "errors": list(state.get("errors", [])) + errors,
            }

        def fuse_node(state: FinanceAgentState) -> dict[str, object]:
            ranked_lists = [
                [FinanceDocument.model_validate(value) for value in values]
                for values in state.get("ranked_lists", [])
            ]
            sparse = [
                FinanceDocument.model_validate(value)
                for value in state.get("sparse_ranked_list", [])
            ]
            fused = reciprocal_rank_fusion(
                [*ranked_lists, sparse],
                k=self._settings.rrf_k,
                limit=self._settings.rrf_candidates,
            )
            return {"candidates": [document.model_dump() for document in fused]}

        async def rerank_node(state: FinanceAgentState) -> dict[str, object]:
            candidates = [
                FinanceDocument.model_validate(value)
                for value in state.get("candidates", [])
            ]
            ranked = await self._reranker.rerank(
                str(state["user_query"]), candidates, limit=self._settings.rerank_top_n
            )
            return {"reranked_documents": [document.model_dump() for document in ranked]}

        async def evaluate_node(state: FinanceAgentState) -> dict[str, object]:
            current = [
                FinanceDocument.model_validate(value)
                for value in state.get("reranked_documents", [])
            ]
            evaluation = await self._evaluator.evaluate(str(state["user_query"]), current)
            prior = ContextEvaluation.model_validate(state["best_evaluation"]) if state.get("best_evaluation") else None
            choose_current = _should_replace_best(
                prior,
                evaluation,
                tie_tolerance=self._settings.sufficiency_tie_tolerance,
            )
            result: dict[str, object] = {"evaluation": evaluation.model_dump()}
            if choose_current:
                result.update(
                    best_context=[document.model_dump() for document in current],
                    best_evaluation=evaluation.model_dump(),
                )
            return result

        async def reformulate_node(state: FinanceAgentState) -> dict[str, object]:
            evaluation = ContextEvaluation.model_validate(state["evaluation"])
            queries = evaluation.suggested_queries
            if not queries:
                queries = [
                    f"{state['user_query']} {' '.join(evaluation.missing_concepts)}".strip()
                ]
            return {
                "retry_count": int(state.get("retry_count", 0)) + 1,
                "dense_sub_queries": queries[:3],
                "sparse_keywords": list(dict.fromkeys([
                    *state.get("sparse_keywords", []),
                    *evaluation.missing_concepts,
                ])),
            }

        async def generate_node(state: FinanceAgentState) -> dict[str, object]:
            best_eval = ContextEvaluation.model_validate(state["best_evaluation"]) if state.get("best_evaluation") else None
            best_documents = [
                FinanceDocument.model_validate(value)
                for value in state.get("best_context", [])
            ]
            errors = list(state.get("errors", []))
            if errors and not best_documents:
                result = FinanceAgentResult(
                    status="error",
                    errors=errors,
                    answer="Finance education retrieval is currently unavailable.",
                )
            elif not best_documents:
                result = FinanceAgentResult(
                    status="insufficient",
                    answer="I could not find supporting evidence in the finance handbook.",
                )
            elif best_eval is None or not best_eval.sufficient:
                result = FinanceAgentResult(
                    status="insufficient",
                    answer=(
                        "The finance handbook did not provide enough evidence to answer "
                        "every part of this question reliably."
                    ),
                    citations=_citations(best_documents),
                )
            else:
                context = _format_context(best_documents)
                messages = [
                    SystemMessage(
                        content=(
                            "You are an investment-education assistant. Answer the original question "
                            "only from the supplied handbook passages. Do not give personalized "
                            "recommendations or present live/current financial data. Cite page numbers "
                            "and source. If one part lacks evidence, state that limitation."
                        )
                    ),
                    HumanMessage(content=f"Question: {state['user_query']}\n\nEvidence:\n{context}"),
                ]
                try:
                    response = await self._llm.ainvoke(messages)
                    answer = response if isinstance(response, str) else str(getattr(response, "content", ""))
                    if not answer.strip():
                        raise ValueError("Empty generated answer")
                    result = FinanceAgentResult(
                        answer=answer.strip(),
                        citations=_citations(best_documents),
                        status="answered",
                    )
                except Exception as error:
                    logger.exception("Finance answer generation failed.")
                    result = FinanceAgentResult(
                        answer="Finance answer generation failed; no unsupported answer was produced.",
                        citations=_citations(best_documents),
                        status="error",
                        errors=[*errors, type(error).__name__],
                    )
            return {
                "final_result": result.model_dump(),
                "answer": result.answer,
                "status": result.status,
                "citations": result.citations,
                "handoff_intents": [str(intent) for intent in result.handoff_intents],
                "clarification_question": result.clarification_question or "",
            }

        def handoff_node(state: FinanceAgentState) -> dict[str, object]:
            result = FinanceAgentResult(
                status="handoff",
                handoff_intents=list(state.get("handoff_intents", [])),
                answer="",
            )
            return {
                "final_result": result.model_dump(),
                "answer": result.answer,
                "status": result.status,
                "citations": result.citations,
                "handoff_intents": result.handoff_intents,
                "intent": state.get("intent", "finance_qa"),
            }

        workflow = StateGraph(FinanceAgentState)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("clarify", clarify_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("fuse", fuse_node)
        workflow.add_node("rerank", rerank_node)
        workflow.add_node("evaluate", evaluate_node)
        workflow.add_node("reformulate", reformulate_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("handoff", handoff_node)
        workflow.add_edge(START, "analyze")
        workflow.add_conditional_edges("analyze", _route_analysis)
        workflow.add_conditional_edges("clarify", _route_after_clarification)
        workflow.add_edge("retrieve", "fuse")
        workflow.add_edge("fuse", "rerank")
        workflow.add_edge("rerank", "evaluate")
        workflow.add_conditional_edges("evaluate", self._route_evaluation)
        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("generate", END)
        workflow.add_edge("handoff", END)
        return workflow.compile(checkpointer=self._checkpointer)

    def _route_evaluation(self, state: FinanceAgentState) -> Literal["generate", "reformulate"]:
        evaluation = ContextEvaluation.model_validate(state["evaluation"])
        if evaluation.sufficient or int(state.get("retry_count", 0)) >= self._settings.max_retrieval_retries:
            return "generate"
        return "reformulate"


def _deterministic_handoff(query: str) -> list[str]:
    if SCOPE_EXCLUSIONS[0].search(query):
        return ["tax"]
    if SCOPE_EXCLUSIONS[1].search(query) or SCOPE_EXCLUSIONS[2].search(query):
        return ["tax"]
    if SCOPE_EXCLUSIONS[3].search(query):
        return ["market"]
    if SCOPE_EXCLUSIONS[4].search(query):
        return ["portfolio"]
    return []


def _route_analysis(state: FinanceAgentState) -> Literal["handoff", "clarify", "retrieve"]:
    if state.get("handoff_intents"):
        return "handoff"
    analysis = QueryAnalysis.model_validate(state.get("analysis") or {})
    return "clarify" if analysis.needs_clarification else "retrieve"


def _route_after_clarification(state: FinanceAgentState) -> Literal["handoff", "analyze"]:
    return "handoff" if state.get("handoff_intents") else "analyze"


def _should_replace_best(
    previous: ContextEvaluation | None,
    current: ContextEvaluation,
    *,
    tie_tolerance: float,
) -> bool:
    if previous is None:
        return True
    if current.sufficient != previous.sufficient:
        return current.sufficient
    return current.sufficiency_score > previous.sufficiency_score + tie_tolerance


def _citations(documents: list[FinanceDocument]) -> list[str]:
    citations = []
    for document in documents:
        page = document.metadata.get("page_number", "unknown")
        citations.append(f"Financial Education Handbook, page {page} ({document.doc_id})")
    return citations


def _format_context(documents: list[FinanceDocument]) -> str:
    return "\n\n".join(
        f"[Source: {document.source}; page {document.metadata.get('page_number', 'unknown')}; "
        f"doc_id: {document.doc_id}]\n{document.content}"
        for document in documents
    )
