import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ai_finance_assistant.rag.evaluation import ContextEvaluation
from ai_finance_assistant.rag.models import RetrievedDocument
from ai_finance_assistant.rag.tax_education import extract_tax_metadata

logger = logging.getLogger(__name__)

TAX_EVALUATION_PROMPT = """Evaluate retrieved evidence for US tax education; do not answer.
Judge only what the original question requests. A definition needs a definition;
a comparison or multipart question needs evidence for every requested part.
For a general standard-deduction question, evidence describing the standard deduction as
an alternative to itemized deductions and how it is claimed supports a concise conceptual
explanation. Do not require deduction amounts, tax brackets, every exception, or filing
status for that general question. The answer must stay within the supplied evidence.
Respect explicit tax year, filing year, filing status, jurisdiction, and taxpayer context.
When no year is specified, use the corpus scope: tax year 2026, generally filed in 2027.
An omitted year is not missing evidence. Explicit years always override this default;
never substitute 2026 evidence for a different requested year.
For a tax-bracket question, interpret an unqualified income amount as taxable ordinary
income; for example, 'income 130K filing single' means $130,000 taxable income, Single.
Do not require gross-income deductions, age, dependents, or other eligibility details to
locate that amount in a supported marginal bracket. If the user explicitly says gross
income, salary, wages, or take-home pay, do not relabel it taxable income; evidence may
support a conditional explanation instead of an exact determination.
Mark evidence sufficient when it supports a useful answer under these defaults. Minor
ambiguity is not missing evidence. Recommend a focused follow-up only for a very vague
question whose subject or essential inputs cannot be identified; record the needed detail
in the reason. Do not use suggested_queries for user follow-ups: they are retrieval queries.
Do not infer other missing taxpayer details or invent unsupported rules or amounts.
Assess actual passage text, not titles or relevance scores. Identify unsupported amounts,
eligibility conditions, and missing concepts. Corpus references are not live verification;
do not claim that current rules were checked. Return sufficient, missing concepts,
focused suggested queries, and a reason."""


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
