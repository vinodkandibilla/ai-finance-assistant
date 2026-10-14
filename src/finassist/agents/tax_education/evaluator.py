import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from finassist.prompts import TAX_EVALUATION_PROMPT
from finassist.rag.evaluation import ContextEvaluation
from finassist.rag.models import RetrievedDocument
from finassist.rag.tax_education import extract_tax_metadata

logger = logging.getLogger(__name__)


class TaxContextEvaluator:
    def __init__(self, llm: object) -> None:
        self._structured_llm = llm.with_structured_output(ContextEvaluation)

    async def evaluate(self, query: str, documents: list[RetrievedDocument]) -> ContextEvaluation:
        if not documents:
            return ContextEvaluation(
                sufficient=False,
                missing_concepts=["supporting tax evidence"],
                suggested_queries=[query],
                reason="No tax passages were retrieved.",
            )
        constraints = extract_tax_metadata(query)
        # Year-specific evidence cannot be silently relaxed on a retrieval retry.
        for field in ("tax_year", "filing_year"):
            expected = constraints.get(field)
            if expected is not None and any(
                str(document.metadata.get(field)) != str(expected) for document in documents
            ):
                return ContextEvaluation(
                    sufficient=False,
                    missing_concepts=[f"evidence for {field} {expected}"],
                    suggested_queries=[query],
                    reason=f"Retrieved evidence does not consistently match {field} {expected}.",
                )
        evidence = [
            {"content": document.content, "metadata": document.metadata, "source": document.source}
            for document in documents
        ]
        try:
            result = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=TAX_EVALUATION_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "question": query,
                                "explicit_constraints": constraints,
                                "evidence": evidence,
                            }
                        )
                    ),
                ]
            )
            return (
                result
                if isinstance(result, ContextEvaluation)
                else ContextEvaluation.model_validate(result)
            )
        except Exception as error:
            logger.warning("Tax evidence evaluation failed: %s", type(error).__name__)
            return ContextEvaluation(
                sufficient=False,
                missing_concepts=["evidence evaluation"],
                reason=f"Evidence evaluation failed: {type(error).__name__}.",
            )
