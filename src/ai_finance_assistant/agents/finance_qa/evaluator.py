from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ai_finance_assistant.agents.finance_qa.schemas import ContextEvaluation, FinanceDocument

logger = logging.getLogger(__name__)


class FinanceContextEvaluator:
    def __init__(self, llm: object) -> None:
        self._structured_llm = llm.with_structured_output(ContextEvaluation)

    async def evaluate(self, query: str, documents: list[FinanceDocument]) -> ContextEvaluation:
        if not documents:
            return ContextEvaluation(
                sufficient=False,
                sufficiency_score=0,
                missing_concepts=["supporting handbook evidence"],
                suggested_queries=[query],
                reason="No in-scope handbook passages were retrieved.",
            )
        # A literal glossary entry can fully support a simple definition. This
        # shortcut does not apply to comparisons or questions with qualifiers.
        definition = re.fullmatch(
            r"\s*what\s+is\s+(?:an?\s+|the\s+)?([A-Za-z][A-Za-z -]{0,60}?)\s*[?.]?\s*",
            query,
            re.IGNORECASE,
        )
        if definition:
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
                            sufficiency_score=1.0,
                            reason="A handbook glossary entry supports the requested definition.",
                        )
        context = "\n\n".join(
            f"[doc_id={document.doc_id}; source={document.source}; "
            f"page={document.metadata.get('page_number', 'unknown')}]\n{document.content}"
            for document in documents
        )
        prompt = [
            SystemMessage(
                content=(
                    "Evaluate whether the combined passages fully support an educational answer "
                    "to the investment question at the level of detail actually requested. "
                    "For 'What is X?', a direct definition and basic characteristics are enough; "
                    "Require examples, trading mechanics, risks, or comparisons only when asked. "
                    "For comparisons and multipart questions, require evidence for each part. "
                    "Ignore unrelated passages when relevant evidence is present. "
                    "Judge coverage, not just relevance. Do not answer. Return an explicitly "
                    "heuristic score from 0 to 1, missing concepts, and focused search queries."
                )
            ),
            HumanMessage(content=f"Question: {query}\n\nPassages:\n{context}"),
        ]
        try:
            result = await self._structured_llm.ainvoke(prompt)
            return (
                result
                if isinstance(result, ContextEvaluation)
                else ContextEvaluation.model_validate(result)
            )
        except Exception as error:
            logger.exception("Finance evidence evaluation failed.")
            return ContextEvaluation(
                sufficient=False,
                sufficiency_score=0,
                missing_concepts=["evidence evaluation"],
                suggested_queries=[],
                reason=f"Evidence evaluation failed: {type(error).__name__}.",
            )
