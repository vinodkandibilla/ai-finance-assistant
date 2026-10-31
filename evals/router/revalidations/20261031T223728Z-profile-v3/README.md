# Customer profile routing validation

The production-default router prompt `v3` adds customer investment profile retrieval
by customer name/ID. It also distinguishes general investment risk-profile education
from a stored customer profile and from unrelated social profiles. Tool access and
identity enforcement are unchanged.

## Observed results

Model: `gpt-4o-mini`, production classifier settings, 10-second deadline, Market
quotes enabled. The active settings matched the source prompt in this new process.
No guardrail or specialist execution was included and no LangSmith experiment was
published. The run retains the exact prompt and 71-case dataset snapshots.

| Check | Result |
| --- | --- |
| Old v2: View C011 profile | other, mismatched |
| Old v2: Get customer C011 profile | other, mismatched |
| v3 full route dataset | 70/71 matched (98.59%), 51/51 critical, no errors |
| v3: View C011 profile, additional repeats | portfolio → portfolio_analysis_node, 3/3 |
| v3: Get customer C011 profile, additional repeats | portfolio → portfolio_analysis_node, 3/3 |
| General investment risk-profile concept | finance_qa, matched |
| Unrelated Instagram profile | other, matched |

The one remaining full-dataset mismatch is the previously known “How much should I
invest?” case: expected clarify, observed goals. No labels were changed to improve
the score. The old prompt was checked on the four new cases only; this is not a
paired full-dataset A/B comparison. Repetitions are additional diagnostics, not
added to the 71-case denominator.

Offline checks: 163 router/evaluation/classifier tests passed; 43 Portfolio routing,
tool and authorization tests passed. External guardrails/tracing were disabled for
these offline runs. Ruff and whitespace checks passed. Stored A/B snapshots remain
unchanged; B represents v2, not the current v3 source default.

## Apply and inspect

Restart FastAPI to load the new prompt. Remove/update an older explicit
`INTENT_ROUTER_SYSTEM_PROMPT` override if one is configured. Routing a profile query
to Portfolio does not grant access: the selected application's manager/customer
scope still controls whether C011 can be viewed. No live profile disclosure or
end-to-end answer formatting is claimed by this router check.

Files: `prompt.txt`, `cases.jsonl`, `configuration.json`, `v2-profile-cases.json`,
`routes.json`, `profile-repeats.json`, `progress.json`.

Prompt SHA-256: `8a33e99fc61f4051d4ce223aab900f1597b3576040b771b7dc9c95ca99240a4f`.
