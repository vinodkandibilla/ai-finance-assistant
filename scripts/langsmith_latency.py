"""Report latency percentiles for completed LangSmith root traces (read-only).

Usage: .venv/bin/python scripts/langsmith_latency.py --app finpilot-ai-dev --days 7
"""

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langsmith import Client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app",
        "--project",
        dest="project",
        default="finpilot-ai-dev",
        help="App's LangSmith project name (default: finpilot-ai-dev)",
    )
    parser.add_argument("--days", type=int, default=7, help="Lookback in whole days (default: 7)")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be greater than zero")

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    until = datetime.now(UTC)
    since = until - timedelta(days=args.days)
    durations = []
    errors = 0
    skipped = 0
    client = Client()
    for run in client.list_runs(
        project_name=args.project,
        is_root=True,
        start_time=since,
        select=["start_time", "end_time", "error"],
    ):
        if not since <= run.start_time <= until:
            continue
        if run.end_time is None or run.end_time < run.start_time:
            skipped += 1
            continue
        durations.append((run.end_time - run.start_time).total_seconds())
        errors += int(bool(run.error))

    print(f"Project: {args.project}")
    print(f"Window (UTC, request start): {since.isoformat()} to {until.isoformat()}")
    print(f"Completed root traces: {len(durations)} (with errors: {errors})")
    print(f"Skipped incomplete/invalid root traces: {skipped}")
    if not durations:
        print("No completed root traces found; percentiles unavailable.")
        return
    for percentile in (50, 80, 85, 90, 95, 99):
        value = np.percentile(durations, percentile, method="linear")
        print(f"P{percentile}: {value:.2f} seconds")
    print("Includes errors; measures root trace duration, not browser latency.")
    print("Percentiles use linear interpolation; small samples may be unrepresentative.")


if __name__ == "__main__":
    main()
