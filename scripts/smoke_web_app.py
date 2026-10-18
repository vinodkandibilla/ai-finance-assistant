"""Opt-in HTTP smoke checks against independently running FastAPI and Streamlit.

Run from the repository root. --live-agents makes provider-backed requests and may
incur provider usage. Only status metadata is printed, never response bodies.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from finassist.web_app.config import WebSettings
from finassist.web_app.services.api_client import ApiError, FinanceApiClient
from finassist.web_app.services.models import QueryRequest, ResumeRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-agents", action="store_true")
    parser.add_argument(
        "--intent",
        action="append",
        choices=["finance_qa", "portfolio", "market", "goals", "news", "tax"],
    )
    parser.add_argument("--ui-url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    settings = WebSettings.from_environment()
    failed = False
    with httpx.Client(timeout=10) as http:
        response = http.get(settings.api_base_url + "/openapi.json")
        response.raise_for_status()
        actual = response.json()
        reference = json.loads(
            (Path(__file__).resolve().parents[1] / "docs/api/finassist-openapi.json").read_text()
        )
        matches = (
            actual["paths"] == reference["paths"]
            and actual["components"] == reference["components"]
        )
        print(f"OpenAPI reference match: {matches}", flush=True)
        failed |= not matches
        for path in [
            "/_stcore/health",
            "/",
            "/finance_qa",
            "/portfolio",
            "/market",
            "/goals",
            "/news",
            "/tax",
        ]:
            result = http.get(args.ui_url.rstrip("/") + path)
            print(f"UI {path}: HTTP {result.status_code}", flush=True)
            failed |= result.status_code != 200
    with FinanceApiClient(settings) as client:
        print(f"API health: {client.health().status}", flush=True)
        if not args.live_agents:
            return int(failed)
        examples = [
            ("finance_qa", "Explain how an ETF differs from a mutual fund for a beginner."),
            ("portfolio", "Show the portfolio value, holdings and allocation for customer C001."),
            ("market", "What is the latest available price of AAPL?"),
            (
                "goals",
                "Can I reach $10000 in two years with $1000 now and $300 monthly at 0% return?",
            ),
            ("news", "Summarize the latest financial-market news with sources."),
            ("tax", "Explain marginal versus effective US federal income tax rates for 2026."),
            ("goals", "Help me plan a savings goal."),
        ]
        for expected, question in examples:
            if args.intent and expected not in args.intent:
                continue
            try:
                result = client.query(QueryRequest(query=question))
                if result.clarification_question and result.thread_id:
                    print(f"{expected}: clarification received; submitting resume", flush=True)
                    result = client.resume(
                        ResumeRequest(
                            thread_id=result.thread_id,
                            reply=(
                                "Target $10000, current savings $1000, monthly saving $300, "
                                "deadline two years, assumed annual return 0%."
                            ),
                        )
                    )
                calculation_count = len(result.calculations or []) + int(
                    result.calculation is not None
                )
                print(
                    f"{expected}: intent={result.intent}, handler={result.executed_handler}, "
                    f"status={result.status or 'answered'}, sources={len(result.citations or [])}, "
                    f"calculations={calculation_count}",
                    flush=True,
                )
                failed |= not result.answer.strip() or bool(result.clarification_question)
                failed |= result.intent != expected or result.status in {
                    "error",
                    "insufficient",
                    "out_of_scope",
                }
            except ApiError as error:
                print(f"{expected}: {error.code}, HTTP {error.status_code}", flush=True)
                failed = True
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (httpx.HTTPError, ApiError, ValueError):
        print("Smoke check could not complete. Check server availability and configuration.")
        sys.exit(1)
