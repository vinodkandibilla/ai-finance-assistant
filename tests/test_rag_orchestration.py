import pytest

from ai_finance_assistant.rag.orchestration import (
    GenerationPolicy,
    generate_answer,
    route_evaluation,
)

TAX_POLICY = GenerationPolicy(
    system_prompt="Use evidence",
    no_evidence_answer="No evidence",
    insufficient_answer="Insufficient",
    retrieval_error_answer="Retrieval failed",
    generation_error_answer="Generation failed",
    errors_override_insufficient=True,
)


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_retry_limit_and_sufficiency(limit):
    evaluation = {"sufficient": False, "reason": "Missing"}
    assert route_evaluation(evaluation, limit, limit) == "generate"
    if limit:
        assert route_evaluation(evaluation, limit - 1, limit) == "reformulate"
    assert route_evaluation({"sufficient": True, "reason": "Covered"}, 0, limit) == "generate"


@pytest.mark.parametrize("policy", [TAX_POLICY])
@pytest.mark.parametrize("response", ["", "   ", None, ["unsupported content block"]])
async def test_generation_rejects_invalid_answers(policy, response):
    class LLM:
        async def ainvoke(self, messages):
            return response

    result = await generate_answer(
        llm=LLM(),
        query="Question",
        has_documents=True,
        evaluation={"sufficient": True, "reason": "Covered"},
        context="Evidence",
        citations=["Source"],
        errors=["prior retrieval failure"],
        policy=policy,
    )
    assert result["status"] == "error"
    assert result["answer"] == policy.generation_error_answer
    assert result["citations"] == ["Source"]
    assert result["errors"] == ["prior retrieval failure", "ValueError"]


@pytest.mark.parametrize("policy", [TAX_POLICY])
async def test_insufficient_context_never_calls_generation(policy):
    class LLM:
        async def ainvoke(self, messages):
            raise AssertionError("Must not generate without sufficient evidence")

    result = await generate_answer(
        llm=LLM(),
        query="Question",
        has_documents=True,
        evaluation={"sufficient": False, "reason": "Missing"},
        context="Evidence",
        citations=["Source"],
        errors=[],
        policy=policy,
    )
    assert result["status"] == "insufficient"
    assert result["citations"] == ["Source"]
