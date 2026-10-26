# Compact router prompt comparison

First candidate rejected: generic planning regressed. Results retained for transparency.

Reference: `evals/router/baselines/20261021T000647364905Z`. The 63-case dataset, model, timeout, routing setting and mode match the reference.

| Mode | Baseline accuracy | Candidate accuracy | Critical accuracy | Errors |
| --- | --- | --- | --- | --- |
| classifier | 98.18% | 96.36% | 100.00% | 0 |
| routes | 98.41% | 96.83% | 100.00% | 0 |

- classifier: goals_generic expected goals, got clarify
- classifier: clarify_how_much expected clarify, got goals
- routes: goals_generic expected goals, got clarify
- routes: clarify_how_much expected clarify, got goals

The retained revision restores generic personal planning coverage. Original system prompt: 545 words; final: 279 words (~49% reduction). This measures words, not exact input-token savings; tool/schema/request tokens remain. Final offline checks: 197 passed; Ruff passed.

One run per mode: matching scores do not prove identical future behavior. No critical-case regressions in these runs. Prompt and dataset snapshots, configuration hashes, source commit/dirty status and detailed results are retained. The known noncritical investment-amount ambiguity remains. Live calls emitted existing Pydantic serialization warnings but no evaluation errors. UTC directory timestamps correspond to October 20, 2026 in America/Phoenix.
