import json
from pathlib import Path

import pytest

from finassist.agents.common.schemas import AgentResult
from finassist.evaluation.finance.grounding import evaluate_grounding
from finassist.rag.models import RetrievedDocument

DATASET = Path(__file__).resolve().parents[1] / "evals/finance/grounding/cases.jsonl"
ROWS = [json.loads(line) for line in DATASET.read_text().splitlines()]


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_mvp_grounding(row):
    response = AgentResult.model_validate(row["response"])
    evidence = [RetrievedDocument.model_validate(d) for d in row["evidence"]]
    result = evaluate_grounding(
        response,
        evidence,
        manually_supported=row["manually_supported"],
        review_notes=row["review_notes"],
    )
    assert result.status == row["expected_status"]
    assert result.manually_supported == row["manually_supported"]
    assert result.review_notes == row["review_notes"]
    assert len(result.invalid_citations) == row["invalid_count"]


@pytest.mark.parametrize("page", [None, "9", True, 0, -1])
def test_malformed_page_cannot_validate_a_citation(page):
    result = evaluate_grounding(
        AgentResult(
            answer="An answer.",
            citations=["Financial Education Handbook, page 9"],
            status="answered",
        ),
        [RetrievedDocument(content="Evidence", source="fixture", metadata={"page_number": page})],
        manually_supported=True,
    )
    assert result.status == "failed"
    assert result.invalid_citations == ["Financial Education Handbook, page 9"]


class JudgeModel:
    def __init__(self, verdict=None, error=None):
        self.verdict, self.error = verdict, error
        self.messages = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.verdict


def judge_inputs():
    row = ROWS[0]
    return (
        AgentResult.model_validate(row["response"]),
        [RetrievedDocument.model_validate(d) for d in row["evidence"]],
    )


@pytest.mark.parametrize(
    "verdict,status,supported",
    [
        ("supported", "passed", True),
        ("unsupported", "failed", False),
        ("uncertain", "needs_review", None),
    ],
)
async def test_llm_verdict_and_cited_evidence_only(verdict, status, supported):
    from finassist.evaluation.finance.grounding import evaluate_grounding_with_llm

    response, evidence = judge_inputs()
    evidence.append(
        RetrievedDocument(
            content="Uncited distractor", source="fixture", metadata={"page_number": 10}
        )
    )
    model = JudgeModel({"verdict": verdict, "reason": "Reviewed against passages"})
    result = await evaluate_grounding_with_llm(response, evidence, model=model)
    assert result.status == status
    assert result.judge_supported is supported
    assert result.manually_supported is None
    assert result.judge_reason == "Reviewed against passages"
    payload = json.loads(model.messages[1].content)
    assert len(payload["cited_passages"]) == 1
    assert "Uncited distractor" not in model.messages[1].content


@pytest.mark.parametrize(
    "raw", [None, {}, {"verdict": "invalid", "reason": "x"}, {"verdict": "supported", "reason": ""}]
)
async def test_invalid_judge_output_needs_review(raw):
    from finassist.evaluation.finance.grounding import evaluate_grounding_with_llm

    result = await evaluate_grounding_with_llm(*judge_inputs(), model=JudgeModel(raw))
    assert result.status == "needs_review"
    assert result.judge_supported is None


async def test_provider_failure_needs_review_without_private_details():
    from finassist.evaluation.finance.grounding import evaluate_grounding_with_llm

    result = await evaluate_grounding_with_llm(
        *judge_inputs(), model=JudgeModel(error=RuntimeError("private provider payload"))
    )
    assert result.status == "needs_review"
    assert "private provider payload" not in result.model_dump_json()


async def test_judge_is_not_called_for_invalid_citations():
    from finassist.evaluation.finance.grounding import evaluate_grounding_with_llm

    response, evidence = judge_inputs()
    response.citations = ["Financial Education Handbook, page 999"]
    model = JudgeModel({"verdict": "supported", "reason": "Must not override citation errors"})
    result = await evaluate_grounding_with_llm(response, evidence, model=model)
    assert result.status == "failed"
    assert not model.messages


async def test_judge_timeout_cancels_pending_call():
    import asyncio

    from finassist.evaluation.finance.grounding import evaluate_grounding_with_llm

    class SlowJudge(JudgeModel):
        cancelled = False

        async def ainvoke(self, messages):
            try:
                await asyncio.sleep(60)
            finally:
                self.cancelled = True

    model = SlowJudge()
    result = await evaluate_grounding_with_llm(*judge_inputs(), model=model, timeout_seconds=0.01)
    assert result.status == "needs_review"
    assert model.cancelled
