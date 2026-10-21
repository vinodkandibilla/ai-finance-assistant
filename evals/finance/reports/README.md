# Combined Finance Q&A reports

From the repository root:

```bash
.venv/bin/python scripts/evaluate_finance.py
```

Runs existing offline pytest suites once, then writes a new UTC timestamp folder:

- `summary.md`: quick area-by-area view.
- `results.json`: per-test outcomes, area counts, dataset/corpus hashes, Python
  version, timestamps, source commit and dirty-worktree flag.
- `pytest.xml`: native pytest JUnit details, including failed assertions.
- `pytest.log`: test output, warnings and collection errors.

Use `--output-root /tmp/finance-reports` to change the parent directory. A nonzero
exit code indicates failed, errored, skipped or missing offline coverage. Reports
are retained on test failures. Reports use the source commit plus a dirty flag;
a dirty tree is not a complete source snapshot. Keep the evaluated changes in Git
for reproducible comparisons.

Offline areas are retrieval plumbing/dataset validation, grounding checker tests,
calculations, clarification/scope and failures. Tests shared by areas run only once
but contribute to multiple area counts; do not sum area counts as a unique total.
Some reused runtime suites also cover tax and are identified by test IDs in JSON.

**Offline passes are not live quality scores.** Actual retrieval quality, live
answer grounding and live tool selection are explicitly `not_run`. No model is
used, so model metadata is null. There are no LLM, embedding or external-service
calls. Expected/actual values are available in failed pytest assertions where
provided; this command does not invent predictions for unrun live cases.

For future live integration, add real per-case results for those areas, preserve
model/prompt/index metadata and keep offline and live results separate. A live
runner, judge calibration and retrieval scoring remain pending; do not mark them
passed based on mocked tests.

`report_type=offline_test_summary` explicitly identifies the report scope.
`unique_test_counts` counts the JUnit test records once, independently of area
membership. A passed area is a suite result, never a live accuracy percentage.
See [Finance Q&A evaluation status](../README.md) for implemented and pending work.
Earlier timestamped reports retain their original format; new runs include these
fields. Do not rewrite historical run results to imply new checks were performed.
