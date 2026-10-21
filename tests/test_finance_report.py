import runpy
from pathlib import Path

import pytest

REPORT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_finance.py"))


@pytest.mark.parametrize(
    "node,outcome",
    [
        ("", "passed"),
        ('<failure message="wrong result"/>', "failed"),
        ('<error message="setup failed"/>', "error"),
        ("<skipped/>", "skipped"),
    ],
)
def test_test_outcomes_are_retained(tmp_path, node, outcome):
    xml = tmp_path / "pytest.xml"
    xml.write_text(
        '<testsuites><testsuite><testcase classname="test_compound_interest" '
        f'name="example" time="0.1">{node}</testcase></testsuite></testsuites>'
    )
    report = REPORT["summarize"](xml, 0 if outcome in {"passed", "skipped"} else 1)
    assert report["report_type"] == "offline_test_summary"
    assert report["unique_test_counts"][outcome] == 1
    assert sum(report["unique_test_counts"].values()) == 1
    assert report["tests"][0]["outcome"] == outcome
    assert report["areas"]["calculations"]["counts"][outcome] == 1
    assert report["areas"]["retrieval_quality"]["status"] == "not_run"
    report["started_at"] = "2026-10-20T00:00:00+00:00"
    assert "not_run" in REPORT["render_summary"](report)


def test_collection_error_cannot_produce_pass(tmp_path):
    xml = tmp_path / "pytest.xml"
    xml.write_text(
        '<testsuite><testcase classname="test_compound_interest" name="ok"/></testsuite>'
    )
    report = REPORT["summarize"](xml, 2)
    assert report["areas"]["calculations"]["status"] == "error"
    assert report["areas"]["failures"]["status"] == "error"
