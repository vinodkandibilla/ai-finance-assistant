import hashlib

import pytest

from finassist.evaluation.router.evaluation import RouterEvalCase, RouterEvalResult
from finassist.evaluation.router.reporting import build_router_report, create_run_metadata


def result(index, status, critical=True, mode="classifier"):
    return RouterEvalResult(
        case=RouterEvalCase(
            id=str(index),
            query="Explain ETFs",
            expected_classified_intent="finance_qa",
            expected_handler="finance_qa_node",
            tags=["finance"],
            critical=critical,
        ),
        mode=mode,
        status=status,
        predicted_intent="tax" if status == "failed" else None,
        error_category="timeout" if status == "error" else None,
    )


def metadata(results, **updates):
    return create_run_metadata(
        [r.case for r in results],
        **dict(
            mode="classifier",
            model="offline-stub",
            system_prompt="Classify",
            router_timeout_seconds=10,
            market_quotes_enabled=True,
        )
        | updates,
    )


def test_scores_errors_skips_critical_and_issue_details():
    results = [
        result(1, "passed"),
        result(2, "failed", False),
        result(3, "error"),
        result(4, "skipped"),
    ]
    report = build_router_report(results, metadata(results))
    assert report.scores.accuracy == pytest.approx(1 / 3)
    assert report.scores.critical_accuracy == 0.5
    assert report.scores.evaluated == 3
    assert report.scores.critical_evaluated == 2
    assert (
        report.scores.passed,
        report.scores.failed,
        report.scores.error,
        report.scores.skipped,
    ) == (1, 1, 1, 1)
    assert [i.case_id for i in report.issues] == ["2", "3"]
    assert report.issues[0].actual_intent == "tax"
    assert report.issues[0].expected_handler is None
    assert report.issues[1].error_category == "timeout"
    assert report.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("results", [[], [result(1, "skipped")]])
def test_no_evaluated_cases_have_null_accuracy(results):
    report = build_router_report(results, metadata(results))
    assert report.scores.accuracy is None
    assert report.scores.critical_accuracy is None


def test_no_critical_cases_have_null_critical_accuracy():
    results = [result(1, "passed", False)]
    scores = build_router_report(results, metadata(results)).scores
    assert scores.accuracy == 1
    assert scores.critical_accuracy is None


def test_metadata_hashes_and_configuration():
    results = [result(1, "passed")]
    original = metadata(results)
    assert original.prompt_sha256 == hashlib.sha256(b"Classify").hexdigest()
    assert original.started_at.utcoffset().total_seconds() == 0
    assert original.dataset_sha256 == metadata(results).dataset_sha256
    assert original.prompt_sha256 != metadata(results, system_prompt="New").prompt_sha256
    assert original.dataset_sha256 != metadata([result(2, "passed")]).dataset_sha256
    assert original.model == "offline-stub"
    assert "api_key" not in original.model_dump_json()


def test_report_rejects_mixed_modes_duplicate_ids_and_different_selection():
    results = [result(1, "passed")]
    with pytest.raises(ValueError, match="mode"):
        build_router_report(results, metadata(results, mode="routes"))
    with pytest.raises(ValueError, match="Duplicate"):
        build_router_report(results * 2, metadata(results * 2))
    with pytest.raises(ValueError, match="selection"):
        build_router_report([], metadata(results))


def test_route_issue_includes_handler_mismatch():
    r = result(1, "failed", mode="routes")
    r.predicted_intent = "finance_qa"
    r.predicted_handler = "clarify_node"
    report = build_router_report([r], metadata([r], mode="routes"))
    assert report.scores.accuracy == 0
    assert report.issues[0].expected_handler == "finance_qa_node"
    assert report.issues[0].actual_handler == "clarify_node"
