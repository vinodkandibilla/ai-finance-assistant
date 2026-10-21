"""On-demand offline Finance Q&A report. No LLM or embedding calls.

Run: .venv/bin/python scripts/evaluate_finance.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "retrieval_checks": [
        "test_finance_retrieval_dataset",
        "test_finance_rag_tool",
        "test_finance_evidence_recovery",
        "test_rag_tool_pipeline",
    ],
    "grounding_checks": ["test_finance_grounding_evaluation"],
    "calculations": [
        "test_compound_interest",
        "test_simple_interest",
        "test_calculation_agent",
        "test_simple_interest_agent",
    ],
    "clarification_scope": [
        "test_finance_autonomous_agent",
        "test_calculation_agent",
        "test_simple_interest_agent",
    ],
    "failures": [
        "test_finance_autonomous_agent",
        "test_agent_deadlines",
        "test_finance_evidence_recovery",
        "test_rag_tool_pipeline",
    ],
}
LIVE_NOTES = {
    "retrieval_quality": "25 labeled questions ready; actual retrieval scoring/baseline pending.",
    "grounding_quality": "LLM judge implemented; live answers and judge calibration not run.",
    "live_tool_selection": "Deferred for MVP; scripted calls do not measure LLM tool/input choice.",
}


def summarize(xml_path: Path, exit_code: int) -> dict:
    tests = []
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        problem = case.find("failure")
        outcome = "failed" if problem is not None else "passed"
        if case.find("error") is not None:
            problem, outcome = case.find("error"), "error"
        if case.find("skipped") is not None:
            outcome = "skipped"
        tests.append(
            {
                "id": f"{case.get('classname', '')}::{case.get('name', '')}",
                "suite": case.get("classname", "").split(".")[-1],
                "outcome": outcome,
                "seconds": float(case.get("time", "0")),
                "reason": problem.get("message", "") if problem is not None else None,
            }
        )
    areas = {}
    for area, suites in GROUPS.items():
        selected = [t for t in tests if t["suite"] in suites]
        counts = {
            s: sum(t["outcome"] == s for t in selected)
            for s in ("passed", "failed", "error", "skipped")
        }
        if exit_code not in (0, 1) or not selected:
            status = "error"
        elif counts["failed"] or counts["error"]:
            status = "failed"
        elif counts["skipped"]:
            status = "incomplete"
        else:
            status = "passed"
        areas[area] = {"kind": "offline", "status": status, "counts": counts}
    areas.update(
        {
            name: {"kind": "live", "status": "not_run", "note": note}
            for name, note in LIVE_NOTES.items()
        }
    )
    return {
        "report_type": "offline_test_summary",
        "pytest_exit_code": exit_code,
        "unique_test_counts": {
            status: sum(t["outcome"] == status for t in tests)
            for status in ("passed", "failed", "error", "skipped")
        },
        "areas": areas,
        "tests": tests,
    }


def render_summary(report: dict) -> str:
    lines = [
        "# Finance Q&A offline checks",
        "",
        f"Started (UTC): {report['started_at']}",
        "",
        "Offline checks only. No live quality scores or provider calls.",
        "A passed area means its selected test suites passed; it is not an accuracy score.",
        "",
        "Unique test outcomes: "
        + ", ".join(f"{key}: {value}" for key, value in report["unique_test_counts"].items()),
        "",
        "| Area | Kind | Status | Details |",
        "| --- | --- | --- | --- |",
    ]
    for area, data in report["areas"].items():
        details = data.get("note") or ", ".join(f"{k}: {v}" for k, v in data["counts"].items())
        lines.append(f"| {area} | {data['kind']} | {data['status']} | {details} |")
    lines += [
        "",
        "Area counts overlap because some existing tests cover multiple areas.",
        "Some reused runtime suites also cover tax; this is not a finance-only case count.",
        "See results.json for individual test outcomes, pytest.xml for assertions,",
        "and pytest.log for collection errors and warnings.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "evals/finance/reports")
    args = parser.parse_args()
    started = datetime.now(UTC)
    out = args.output_root / started.strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    suites = sorted({suite for group in GROUPS.values() for suite in group})
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *(f"tests/{s}.py" for s in suites),
        f"--junitxml={out.resolve() / 'pytest.xml'}",
    ]
    print("Running offline Finance Q&A checks...", flush=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out / "pytest.log").write_text(completed.stdout + completed.stderr)
    if (out / "pytest.xml").exists():
        report = summarize(out / "pytest.xml", completed.returncode)
    else:
        report = {
            "report_type": "offline_test_summary",
            "unique_test_counts": {s: 0 for s in ("passed", "failed", "error", "skipped")},
            "pytest_exit_code": completed.returncode,
            "tests": [],
            "areas": {
                name: {"kind": "offline", "status": "error", "note": "No test report produced"}
                for name in GROUPS
            },
        }
        report["areas"].update(
            {
                name: {"kind": "live", "status": "not_run", "note": note}
                for name, note in LIVE_NOTES.items()
            }
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    inputs = sorted((ROOT / "evals/finance").glob("*/cases.jsonl"))
    inputs += list((ROOT / "evals/finance/retrieval").glob("corpus.json"))
    report.update(
        started_at=started.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        python=sys.version.split()[0],
        source_commit=revision.stdout.strip() or None,
        working_tree_dirty=bool(dirty.stdout.strip()),
        model=None,
        model_note="No live model used",
        input_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.md").write_text(render_summary(report))
    print(f"Report: {out / 'summary.md'}", flush=True)
    return (
        0
        if all(a["status"] == "passed" for a in report["areas"].values() if a["kind"] == "offline")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
