"""Scores and reproducibility metadata for router evaluation runs."""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from finassist.evaluation.router.evaluation import RouterEvalCase, RouterEvalResult


class RouterRunMetadata(BaseModel):
    started_at: datetime
    mode: Literal["classifier", "routes"]
    model: str
    prompt_version: str = "v1"
    dataset_sha256: str
    router_timeout_seconds: float
    market_quotes_enabled: bool


class RouterScores(BaseModel):
    accuracy: float | None
    critical_accuracy: float | None
    evaluated: int
    critical_evaluated: int
    passed: int
    failed: int
    error: int
    skipped: int


class RouterIssue(BaseModel):
    case_id: str
    critical: bool
    status: str
    expected_intent: str | None
    actual_intent: str | None
    expected_handler: str | None
    actual_handler: str | None
    error_category: str | None


class RouterReport(BaseModel):
    metadata: RouterRunMetadata
    scores: RouterScores
    issues: list[RouterIssue]
    results: list[RouterEvalResult]


def _dataset_hash(cases: Sequence[RouterEvalCase]) -> str:
    # Hash exactly the evaluated selection, independent of JSONL whitespace.
    payload = json.dumps([c.model_dump() for c in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_run_metadata(
    cases: Sequence[RouterEvalCase],
    *,
    mode: Literal["classifier", "routes"],
    model: str,
    router_timeout_seconds: float,
    market_quotes_enabled: bool,
    prompt_version: str = "v1",
) -> RouterRunMetadata:
    """Capture before running; supply the actual classifier and routing configuration."""
    return RouterRunMetadata(
        started_at=datetime.now(UTC),
        mode=mode,
        model=model,
        prompt_version=prompt_version,
        dataset_sha256=_dataset_hash(cases),
        router_timeout_seconds=router_timeout_seconds,
        market_quotes_enabled=market_quotes_enabled,
    )


def build_router_report(
    results: Sequence[RouterEvalResult], metadata: RouterRunMetadata
) -> RouterReport:
    """Exclude skips from denominators; count errors as unsuccessful evaluations."""
    if any(r.mode != metadata.mode for r in results):
        raise ValueError("Result mode does not match run metadata")
    if len({r.case.id for r in results}) != len(results):
        raise ValueError("Duplicate result case IDs")
    if _dataset_hash([r.case for r in results]) != metadata.dataset_sha256:
        raise ValueError("Results do not match the metadata dataset selection/order")
    evaluated = [r for r in results if r.status != "skipped"]
    critical = [r for r in evaluated if r.case.critical]
    passed = sum(r.status == "passed" for r in evaluated)
    scores = RouterScores(
        accuracy=passed / len(evaluated) if evaluated else None,
        critical_accuracy=(
            sum(r.status == "passed" for r in critical) / len(critical) if critical else None
        ),
        evaluated=len(evaluated),
        critical_evaluated=len(critical),
        passed=passed,
        failed=sum(r.status == "failed" for r in results),
        error=sum(r.status == "error" for r in results),
        skipped=sum(r.status == "skipped" for r in results),
    )
    issues = [
        RouterIssue(
            case_id=r.case.id,
            critical=r.case.critical,
            status=r.status,
            expected_intent=r.case.expected_classified_intent,
            actual_intent=r.predicted_intent,
            expected_handler=r.case.expected_handler if r.mode == "routes" else None,
            actual_handler=r.predicted_handler,
            error_category=r.error_category,
        )
        for r in results
        if r.status in {"failed", "error"}
    ]
    return RouterReport(metadata=metadata, scores=scores, issues=issues, results=list(results))
