from pathlib import Path

import pytest

from finassist.core.config import get_settings
from finassist.evaluation.router.evaluation import (
    RouterEvalCase,
    evaluate_classifier,
    evaluate_routes,
    load_router_cases,
)
from finassist.workflow.classifier import (
    ClassificationOutputError,
    ClassificationProviderError,
    ClassificationTimeoutError,
)

DATASET = Path(__file__).resolve().parents[1] / "evals/router/cases.jsonl"


class Classifier:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    async def classify(self, query):
        self.calls.append(query)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def case(**updates):
    return RouterEvalCase.model_validate(
        dict(
            id="example",
            query="What is an ETF?",
            expected_classified_intent="finance_qa",
            expected_handler="finance_qa_node",
            tags=["finance"],
            critical=True,
        )
        | updates
    )


async def test_classifier_matches_mismatches_and_skips_greeting():
    greeting = case(query="Hi", expected_classified_intent=None, expected_handler=None)
    classifier = Classifier("finance_qa", "tax")
    results = await evaluate_classifier([case(), case(), greeting], classifier)
    assert [r.status for r in results] == ["passed", "failed", "skipped"]
    assert len(classifier.calls) == 2
    assert results[1].predicted_intent == "tax"
    assert all(r.handler_matches is None for r in results)
    assert results[0].case.critical
    assert results[0].elapsed_ms >= 0


async def test_dataset_routes_with_labeled_classifier(monkeypatch):
    monkeypatch.setattr(get_settings(), "market_quotes_enabled", True)
    cases = load_router_cases(DATASET)
    classifier = Classifier(
        *(c.expected_classified_intent for c in cases if c.expected_classified_intent is not None)
    )
    results = await evaluate_routes(cases, classifier)
    assert all(r.status == "passed" for r in results)
    assert classifier.calls == [c.query for c in cases if c.expected_classified_intent]
    generic = next(r for r in results if r.case.id == "market_generic_price")
    assert generic.predicted_intent == "market"
    assert generic.predicted_handler == "clarify_node"


async def test_routes_preserve_actual_configuration(monkeypatch):
    monkeypatch.setattr(get_settings(), "market_quotes_enabled", False)
    generic = case(
        query="What is the stock price?",
        expected_classified_intent="market",
        expected_handler="clarify_node",
    )
    (result,) = await evaluate_routes([generic], Classifier("market"))
    assert result.status == "failed"
    assert result.intent_matches is True
    assert result.handler_matches is False
    assert result.predicted_handler == "market_analysis_node"


@pytest.mark.parametrize("evaluate", [evaluate_classifier, evaluate_routes])
@pytest.mark.parametrize(
    "error,category",
    [
        (ClassificationTimeoutError, "timeout"),
        (ClassificationProviderError, "provider_unavailable"),
        (ClassificationOutputError, "invalid_output"),
    ],
)
async def test_errors_are_recorded_and_evaluation_continues(evaluate, error, category):
    results = await evaluate([case(), case()], Classifier(error("private"), "finance_qa"))
    assert results[0].status == "error"
    assert results[0].error_category == category
    assert results[0].intent_matches is None
    assert "private" not in results[0].model_dump_json()
    assert results[1].status == "passed"


async def test_unexpected_errors_propagate():
    with pytest.raises(RuntimeError):
        await evaluate_classifier([case()], Classifier(RuntimeError("bug")))


@pytest.mark.parametrize(
    "record",
    [
        "not json",
        "{}",
        '{"id": "x"}',
        case().model_dump_json().replace('"critical":true', '"critical":"true"'),
        case().model_dump_json().replace('"finance_qa"', '"invalid"'),
        case().model_dump_json().replace('"finance_qa_node"', "null"),
        case().model_dump_json().replace('["finance"]', '[" "]'),
    ],
)
def test_loader_reports_invalid_line(tmp_path, record):
    path = tmp_path / "cases.jsonl"
    path.write_text(case().model_dump_json() + "\n" + record)
    with pytest.raises(ValueError, match=r":2: invalid router case"):
        load_router_cases(path)


def test_loader_rejects_duplicates_and_empty_dataset(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text((case().model_dump_json() + "\n") * 2)
    with pytest.raises(ValueError, match=r":2: duplicate case ID"):
        load_router_cases(path)
    path.write_text("")
    with pytest.raises(ValueError, match="dataset is empty"):
        load_router_cases(path)
