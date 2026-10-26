# ruff: noqa: E402
"""Run one router prompt against the fixed cases and optionally publish to LangSmith.

Examples:
  .venv/bin/python scripts/evaluate_router_langsmith.py --variant A --mode routes
  .venv/bin/python scripts/evaluate_router_langsmith.py --variant B --mode routes --publish
"""

import argparse
import asyncio
import hashlib
import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langsmith import Client

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from finassist.core.config import Settings
from finassist.evaluation.router.evaluation import (
    evaluate_classifier,
    evaluate_routes,
    load_router_cases,
)
from finassist.evaluation.router.reporting import build_router_report, create_run_metadata
from finassist.workflow.classifier import create_intent_classifier

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
PROMPTS = {
    "A": ROOT / "evals/router/prompts/baseline.txt",
    "B": ROOT / "evals/router/prompts/candidate.txt",
}
PROMPT_VERSIONS = {"A": "v1", "B": "v2"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(PROMPTS), required=True)
    parser.add_argument("--mode", choices=("classifier", "routes"), default="routes")
    parser.add_argument(
        "--publish", action="store_true", help="Upload dataset and create an Experiment"
    )
    parser.add_argument("--dataset-name", default="finassist-router-cases")
    parser.add_argument("--project", default=None, help="LangSmith project name")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=5, help="Number of cases (default: 5)")
    scope.add_argument("--all", action="store_true", help="Run every case in cases.jsonl")
    return parser.parse_args()


def ensure_dataset(client: Client, name: str, cases) -> object:
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = {
            example.metadata.get("case_id")
            for example in client.list_examples(dataset_id=dataset.id)
        }
    else:
        dataset = client.create_dataset(
            name,
            description="FinAssist router evaluation cases",
            metadata={"dataset_sha256": hashlib.sha256(
                json.dumps([case.model_dump() for case in cases], sort_keys=True).encode()
            ).hexdigest()},
        )
        existing = set()
    for case in cases:
        if case.id in existing:
            continue
        client.create_example(
            inputs={"query": case.query},
            outputs={
                "intent": case.expected_classified_intent,
                "handler": case.expected_handler,
            },
            metadata={"case_id": case.id, "tags": case.tags, "critical": case.critical},
            dataset_id=dataset.id,
        )
    client.flush()
    return dataset


def publish_experiment(client: Client, dataset, report, *, variant: str) -> object:
    """Create one LangSmith experiment from the already-computed report."""
    by_query = {result.case.query: result for result in report.results}
    case_ids = {result.case.id for result in report.results}
    examples = [
        example
        for example in client.list_examples(dataset_id=dataset.id)
        if example.metadata.get("case_id") in case_ids
    ]
    if len(examples) != len(report.results):
        raise RuntimeError("LangSmith dataset is missing one or more selected case IDs")

    def lookup(query: str):
        result = by_query.get(query)
        if result is None and "\nQuestion: " in query:
            result = by_query.get(query.rsplit("\nQuestion: ", 1)[1])
        if result is None:
            raise KeyError(f"No local result for dataset query: {query!r}")
        return result

    def target(inputs: dict) -> dict:
        result = lookup(inputs["query"])
        return {
            "intent": result.predicted_intent,
            "handler": result.predicted_handler,
            "status": result.status,
        }

    def score_intent(run, example) -> dict:
        return {
            "key": "intent_correct",
            "score": (run.outputs or {}).get("intent") == (example.outputs or {}).get("intent"),
        }

    def score_handler(run, example) -> dict:
        expected = (example.outputs or {}).get("handler")
        if expected is None:
            return {"key": "handler_correct", "score": None}
        return {
            "key": "handler_correct",
            "score": (run.outputs or {}).get("handler") == expected,
        }

    def score_route(run, example) -> dict:
        outputs, expected = run.outputs or {}, example.outputs or {}
        return {
            "key": "route_correct",
            "score": outputs.get("intent") == expected.get("intent")
            and outputs.get("handler") == expected.get("handler"),
        }

    return client.evaluate(
        target,
        data=examples,
        evaluators=[score_intent, score_handler, score_route],
        experiment_prefix=f"router-{variant}",
        metadata=report.metadata.model_dump(mode="json") | {
            "variant": variant,
            "prompt_version": PROMPT_VERSIONS[variant],
        },
        description=(
            f"FinAssist router {report.metadata.mode} evaluation "
            f"({len(report.results)} cases)"
        ),
        blocking=True,
    )


async def main() -> None:
    args = arguments()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    settings = Settings()
    prompt = PROMPTS[args.variant].read_text(encoding="utf-8")
    all_cases = load_router_cases(ROOT / "evals/router/cases.jsonl")
    cases = all_cases if args.all else all_cases[: args.limit]
    classifier = create_intent_classifier(settings, system_prompt=prompt)
    results = (
        await evaluate_classifier(cases, classifier)
        if args.mode == "classifier"
        else await evaluate_routes(cases, classifier)
    )
    metadata = create_run_metadata(
        cases,
        mode=args.mode,
        model=settings.openai_model,
        prompt_version=PROMPT_VERSIONS[args.variant],
        router_timeout_seconds=settings.router_timeout_seconds,
        market_quotes_enabled=settings.market_quotes_enabled,
    )
    report = build_router_report(results, metadata)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    if args.publish:
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            raise SystemExit("LANGSMITH_API_KEY or LANGCHAIN_API_KEY is required for --publish")
        client = Client(api_key=api_key)
        dataset = ensure_dataset(client, args.dataset_name, cases)
        experiment = publish_experiment(client, dataset, report, variant=args.variant)
        project = args.project or os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
        print(
            f"Published variant {args.variant} experiment "
            f"{experiment.experiment_name!r} to dataset {dataset.name!r}; "
            f"project={project or 'default'!r}."
        )


if __name__ == "__main__":
    asyncio.run(main())
