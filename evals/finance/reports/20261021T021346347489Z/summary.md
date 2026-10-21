# Finance Q&A offline checks

Started (UTC): 2026-10-21T02:13:46.347489+00:00

Offline checks only. No live quality scores or provider calls.
A passed area means its selected test suites passed; it is not an accuracy score.

Unique test outcomes: passed: 129, failed: 0, error: 0, skipped: 0

| Area | Kind | Status | Details |
| --- | --- | --- | --- |
| retrieval_checks | offline | passed | passed: 15, failed: 0, error: 0, skipped: 0 |
| grounding_checks | offline | passed | passed: 25, failed: 0, error: 0, skipped: 0 |
| calculations | offline | passed | passed: 63, failed: 0, error: 0, skipped: 0 |
| clarification_scope | offline | passed | passed: 35, failed: 0, error: 0, skipped: 0 |
| failures | offline | passed | passed: 37, failed: 0, error: 0, skipped: 0 |
| retrieval_quality | live | not_run | 25 labeled questions ready; actual retrieval scoring/baseline pending. |
| grounding_quality | live | not_run | LLM judge implemented; live answers and judge calibration not run. |
| live_tool_selection | live | not_run | Deferred for MVP; scripted calls do not measure LLM tool/input choice. |

Area counts overlap because some existing tests cover multiple areas.
Some reused runtime suites also cover tax; this is not a finance-only case count.
See results.json for individual test outcomes, pytest.xml for assertions,
and pytest.log for collection errors and warnings.
