"""MVP citation validation with manual or on-demand LLM semantic review.
See evals/finance/grounding/README.md for the contract and deferred work.
"""

import asyncio
import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from finassist.agents.common.schemas import AgentResult
from finassist.prompts import GROUNDING_JUDGE_PROMPT
from finassist.rag.models import RetrievedDocument


class GroundingResult(BaseModel):
    status: Literal["passed", "failed", "needs_review"]
    invalid_citations: list[str]
    issues: list[str]
    manually_supported: bool | None
    review_notes: str
    judge_supported: bool | None = None
    judge_reason: str | None = None


def _evidence_citation(document: RetrievedDocument) -> str | None:
    page = document.metadata.get("page_number")
    if type(page) is int and page > 0:
        return f"Financial Education Handbook, page {page}"
    return None


def _check_citations(response: AgentResult, evidence: list[RetrievedDocument]) -> GroundingResult:
    """Check structural prerequisites without making a semantic review decision."""
    valid = {citation for doc in evidence if (citation := _evidence_citation(doc))}
    invalid = sorted(set(response.citations) - valid)
    issues = [f"Citation absent from retrieved evidence: {cite}" for cite in invalid]
    if response.status == "answered" and not response.citations:
        issues.append("Answered response has no citations")
    if not response.answer.strip():
        issues.append("Response has no answer text")
    return GroundingResult(
        status="failed" if issues else "needs_review",
        invalid_citations=invalid,
        issues=issues,
        manually_supported=None,
        review_notes="",
    )


def _outside_scope(response: AgentResult, result: GroundingResult) -> bool:
    if response.status in {"answered", "insufficient"}:
        return False
    result.issues.append("Response status is outside grounding evaluation scope")
    return True


def evaluate_grounding(
    response: AgentResult,
    evidence: list[RetrievedDocument],
    *,
    manually_supported: bool | None = None,
    review_notes: str = "",
) -> GroundingResult:
    """Record human review of all claims against cited passages, after citation checks.

    For an insufficient response, review that it abstains without unsupported facts.
    Valid citations alone never establish semantic support.
    """
    result = _check_citations(response, evidence)
    result.manually_supported = manually_supported
    result.review_notes = review_notes
    if manually_supported is False:
        result.status = "failed"
        result.issues.append("Human review found unsupported or partially supported content")
    if result.status == "failed" or _outside_scope(response, result):
        return result
    if manually_supported is None:
        result.issues.append("Semantic support requires human review")
    else:
        result.status = "passed"
    return result


class GroundingVerdict(BaseModel):
    verdict: Literal["supported", "unsupported", "uncertain"]
    reason: str = Field(min_length=1)


async def evaluate_grounding_with_llm(
    response: AgentResult,
    evidence: list[RetrievedDocument],
    *,
    model: BaseChatModel | None = None,
    timeout_seconds: float = 10.0,
) -> GroundingResult:
    """Check citations first, then judge the whole answer using cited passages only.

    Calling without a model creates ChatOpenAI using the configured finance
    evaluator model and API key. This makes billable calls; tests inject a model.
    Judge uncertainty, invalid output and failures never produce a pass.
    """
    if timeout_seconds <= 0:
        raise ValueError("Judge timeout must be positive")
    result = _check_citations(response, evidence)
    if result.status == "failed" or _outside_scope(response, result):
        return result
    passages = [
        {"citation": citation, "content": doc.content}
        for doc in evidence
        if (citation := _evidence_citation(doc)) in response.citations
    ]
    try:
        async with asyncio.timeout(timeout_seconds):
            if model is None:
                from langchain_openai import ChatOpenAI

                from finassist.core.config import get_settings

                settings = get_settings()
                if not settings.openai_api_key:
                    raise ValueError("Missing judge API key")
                model = ChatOpenAI(
                    model=settings.finance_evaluator_model,
                    api_key=settings.openai_api_key.get_secret_value(),
                    timeout=timeout_seconds,
                    max_retries=0,
                )
            raw = await model.with_structured_output(GroundingVerdict).ainvoke(
                [
                    SystemMessage(content=GROUNDING_JUDGE_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "answer": response.answer,
                                "status": response.status,
                                "cited_passages": passages,
                            }
                        )
                    ),
                ]
            )
            verdict = (
                raw if isinstance(raw, GroundingVerdict) else GroundingVerdict.model_validate(raw)
            )
    except Exception:
        result.issues = ["LLM grounding judge failed or timed out; review required"]
        return result
    result.judge_reason = verdict.reason
    result.judge_supported = {"supported": True, "unsupported": False, "uncertain": None}[
        verdict.verdict
    ]
    result.status = (
        "passed"
        if result.judge_supported is True
        else "failed"
        if result.judge_supported is False
        else "needs_review"
    )
    result.issues = {
        "supported": [],
        "unsupported": ["LLM judge found unsupported content"],
        "uncertain": ["LLM judge was uncertain; review required"],
    }[verdict.verdict]
    return result
