from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from finassist.agents.finance_qa.schemas import ContextEvaluation, FinanceDocument
from finassist.prompts import FINANCE_QA_EVALUATION_PROMPT

logger = logging.getLogger(__name__)


class FinanceContextEvaluator:
    def __init__(self, llm: object) -> None:
        # Configure the LLM to return a structured evidence evaluation.
        self._structured_llm = llm.with_structured_output(ContextEvaluation)

    async def evaluate(self, query: str, documents: list[FinanceDocument]) -> ContextEvaluation:
        if not documents:
            return self._evaluate_missing_evidence(query)

        glossary_evaluation = self._evaluate_glossary_definition(query, documents)
        if glossary_evaluation is not None:
            return glossary_evaluation

        context = self._format_evidence_context(documents)
        prompt = self._build_evaluation_prompt(query, context)
        return await self._evaluate_evidence_with_llm(prompt)

    @staticmethod
    def _evaluate_missing_evidence(query: str) -> ContextEvaluation:
        return ContextEvaluation(
            sufficient=False,
            missing_concepts=["supporting handbook evidence"],
            suggested_queries=[query],
            reason="No in-scope handbook passages were retrieved.",
        )

    @staticmethod
    def _evaluate_glossary_definition(
        query: str, documents: list[FinanceDocument]
    ) -> ContextEvaluation | None:
        # Only simple definition questions qualify for the glossary shortcut.
        definition = re.fullmatch(
            r"\s*what\s+is\s+(?:an?\s+|the\s+)?([A-Za-z][A-Za-z -]{0,60}?)\s*[?.]?\s*",
            query,
            re.IGNORECASE,
        )
        if definition is None:
            return None

        term = definition.group(1).strip().casefold()
        for document in documents:
            lines = document.content.splitlines()
            for index, line in enumerate(lines[:-1]):
                heading = line.strip()
                acronym = re.fullmatch(r"[^()]+\(([A-Z][A-Z0-9.-]+)\)", heading)
                if heading.casefold() != term and not (
                    acronym and acronym.group(1).casefold() == term
                ):
                    continue
                explanation = lines[index + 1].strip()
                if len(explanation.split()) >= 6 and explanation.endswith("."):
                    return ContextEvaluation(
                        sufficient=True,
                        reason="A handbook glossary entry supports the requested definition.",
                    )
        return None

    @staticmethod
    def _format_evidence_context(documents: list[FinanceDocument]) -> str:
        return "\n\n".join(
            f"[doc_id={document.doc_id}; source={document.source}; "
            f"page={document.metadata.get('page_number', 'unknown')}]\n{document.content}"
            for document in documents
        )

    @staticmethod
    def _build_evaluation_prompt(query: str, context: str) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content=FINANCE_QA_EVALUATION_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nPassages:\n{context}"),
        ]

    async def _evaluate_evidence_with_llm(
        self, prompt: list[SystemMessage | HumanMessage]
    ) -> ContextEvaluation:
        try:
            result = await self._structured_llm.ainvoke(prompt)
            return (
                result
                if isinstance(result, ContextEvaluation)
                else ContextEvaluation.model_validate(result)
            )
        except Exception as error:
            logger.exception("Finance evidence evaluation failed.")
            return self._evaluation_failure_result(error)

    @staticmethod
    def _evaluation_failure_result(error: Exception) -> ContextEvaluation:
        return ContextEvaluation(
            sufficient=False,
            missing_concepts=["evidence evaluation"],
            suggested_queries=[],
            reason=f"Evidence evaluation failed: {type(error).__name__}.",
        )
