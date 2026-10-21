"""Reusable router evaluations; callers supply either a live or offline classifier."""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finassist.workflow.builder import _classify_intent
from finassist.workflow.classifier import (
    ClassificationOutputError,
    ClassificationProviderError,
    ClassificationTimeoutError,
    IntentClassifier,
)
from finassist.workflow.state import ExecutedHandler, FinanceIntent


class RouterEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, pattern=r"\S")
    query: str = Field(min_length=1, pattern=r"\S")
    expected_classified_intent: FinanceIntent | None
    expected_handler: ExecutedHandler | None
    tags: list[str] = Field(min_length=1)
    critical: bool
    notes: str | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if (self.expected_classified_intent is None) != (self.expected_handler is None):
            raise ValueError("Greeting labels must both be null")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("Tags must not be blank")
        return self


class RouterEvalResult(BaseModel):
    case: RouterEvalCase
    mode: Literal["classifier", "routes"]
    status: Literal["passed", "failed", "skipped", "error"]
    predicted_intent: FinanceIntent | None = None
    predicted_handler: ExecutedHandler | None = None
    intent_matches: bool | None = None
    handler_matches: bool | None = None
    elapsed_ms: float = 0.0
    error_category: Literal["timeout", "provider_unavailable", "invalid_output"] | None = None


def load_router_cases(path: Path) -> list[RouterEvalCase]:
    """Load JSONL, rejecting invalid records and duplicate IDs with line context."""
    cases: list[RouterEvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            case = RouterEvalCase.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid router case: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case ID {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: dataset is empty")
    return cases


async def evaluate_classifier(
    cases: Sequence[RouterEvalCase], classifier: IntentClassifier
) -> list[RouterEvalResult]:
    """Evaluate only classification; null-labeled greeting cases are skipped."""
    return [await _evaluate_case(case, classifier, mode="classifier") for case in cases]


async def evaluate_routes(
    cases: Sequence[RouterEvalCase], classifier: IntentClassifier
) -> list[RouterEvalResult]:
    """Evaluate production greeting/classification/routing without domain execution."""
    return [await _evaluate_case(case, classifier, mode="routes") for case in cases]


async def _evaluate_case(
    case: RouterEvalCase,
    classifier: IntentClassifier,
    *,
    mode: Literal["classifier", "routes"],
) -> RouterEvalResult:
    if mode == "classifier" and case.expected_classified_intent is None:
        return RouterEvalResult(case=case, mode=mode, status="skipped")

    started = time.perf_counter()
    result = RouterEvalResult(case=case, mode=mode, status="failed")
    try:
        if mode == "classifier":
            result.predicted_intent = await classifier.classify(case.query)
        else:
            state = await _classify_intent({"query": case.query}, classifier=classifier)
            result.predicted_intent = state.get("classified_intent")
            result.predicted_handler = state.get("executed_handler")
            result.handler_matches = result.predicted_handler == case.expected_handler
        result.intent_matches = result.predicted_intent == case.expected_classified_intent
        if result.intent_matches and result.handler_matches is not False:
            result.status = "passed"
    except ClassificationTimeoutError:
        result.status, result.error_category = "error", "timeout"
    except ClassificationProviderError:
        result.status, result.error_category = "error", "provider_unavailable"
    except ClassificationOutputError:
        result.status, result.error_category = "error", "invalid_output"
    finally:
        result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result
