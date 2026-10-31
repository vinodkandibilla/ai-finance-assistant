# Router revalidation — 31 October 2026

This review checks the router after reported clarification questions reached domain
agents. It retains the failures and does not change expected labels or production
routing to improve the score. No LangSmith dataset or experiment was published.

## Scope and configuration

- Current `gpt-4o-mini` classifier with structured output and production sampling defaults.
- Active Settings prompt matches the source default and candidate B (ignoring the
  candidate file's trailing newline); prompt version is `v2`.
- Prompt SHA-256: `9d4ab4b199988e9544fa57386d3fe8951aeaaff076d4917e75e87631298d6ea9`.
- Market quotes enabled, router deadline 10 seconds.
- `routes.json` uses production greeting detection, LLM classification and handler
  selection. It does not execute API guardrails or domain agents.
- Settings were loaded in a new process. This does not establish the prompt loaded
  in an already-running FastAPI process or reproduce the exact UI request in the
  user's trace.
- Fixed benchmark dataset SHA-256:
  `9f7e948224229fab574ede8ecde122d64a9198be91d5ec087aff15f34201f283`.
- Snapshot inputs: `prompt.txt`, `cases.jsonl`, `configuration.json`.

## Fixed benchmark result

| Measure | Result |
| --- | --- |
| Route matches (both classified intent and handler) | 66/67, 98.51% |
| Critical cases | 47/47 |
| Failed cases | 1 |
| Provider/output/timeout errors | 0 |
| Skipped cases | 0 |

The failure is `clarify_how_much`: **“How much should I invest?”** is labeled
`clarify → clarify_node`, but returned `goals → goals_node`. The same case failed
in the retained 21 October baseline and comparisons. An aggregate high score did
not mean this clarification example passed. Its label also needs product review:
“personal planning” in the goals definition can reasonably attract this request.
Do not relabel it merely to raise the score.

The dataset contains only two `clarify` examples. Its overall accuracy therefore
provides limited evidence about clarification behavior. This is one current-prompt
run, not a new A/B comparison or an estimate of production-wide accuracy.

## Repeated clarification checks

These are separate from the fixed benchmark. Counts below use only the five
additional repetitions per query, excluding the initial probe and benchmark.

| Query | Observed classifications | Clarify result |
| --- | --- | --- |
| Which is better? | clarify 5/5 | 5/5 |
| How much should I invest? | goals 5/5 | 0/5 |
| Can you help me with my finances? | other 4/5, finance_qa 1/5 | 0/5 |

The third query is exploratory and is not part of the fixed dataset. Its `clarify`
expectation was a suggested conversational example, not an approved benchmark
label. These results withdraw the earlier claim that it is a reliable clarification
demo. Five repetitions do not guarantee future behavior.

See `clarification-repeats.json`. The user's observed Market result for “Which is
better?” was not reproduced here, either unwrapped or in the Market wrapper.
Compare the exact failing trace input, loaded prompt and process version before
attributing that difference solely to randomness.

## Market UI wrapper check

`scope_market_question` adds Market scope instructions ahead of the actual question.
Three extra live repetitions per query were run with that real helper:

| Wrapped user question | Expected route | Observed route |
| --- | --- | --- |
| Which is better? | clarify → clarify_node | expected route, 3/3 |
| What is the stock price? | market → clarify_node | market → market_analysis_node, 3/3 |

The second mismatch has a deterministic cause. `_is_generic_query` compares the
entire wrapped text against its exact list of generic market questions. That text
no longer equals “what is the stock price”. The specialist can still clarify, but
the router-level clarification rule has been bypassed. The fixed 67-case dataset
tests the unwrapped question and misses this wrapper/clarification combination.
See `market-wrapper-repeats.json`. No production fix was applied in this review.

## Offline regression checks

Run with moderation, injection and LangSmith tracing disabled through process-local
environment variables so offline tests cannot call live guardrail services:

```bash
MODERATION_ENABLED=false INJECTION_CLASSIFIER_ENABLED=false \
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
.venv/bin/python -m pytest -q tests/test_router_decisions.py \
  tests/test_router_evaluation.py tests/test_router_reporting.py
```

**123 passed.** These tests use supplied/stub classifications. They verify routing
and scoring contracts, not live intent selection.

Adding `tests/test_classifier_failures.py`, `tests/test_classifier_timeout.py`,
`tests/test_classifier_logging.py` and `tests/test_greetings.py` produced
**186 passed, 15 failed**. Fourteen failure-response tests and one timeout-response
test expect HTTP 502/503/504; the current API catches those failures and returns
HTTP 200 with a safe `status="error"` response. This is an unresolved API/test
contract mismatch, not a passing suite. Decide the public error contract and align
implementation and tests before claiming full regression success. An initial run
with local guardrail settings enabled also hit network-dependent failures and is
not used as an offline correctness result.

## Evaluation tooling review

The published LangSmith experiments must be interpreted with these runner details:

1. `publish_experiment` replays previously computed predictions through a local
   `target`. Dashboard target latency is replay latency, not the original model
   latency. The actual local evaluation records `elapsed_ms`. Tokens/cost on that
   replay are not evidence of live inference usage.
2. `ensure_dataset` skips an existing case ID without checking its query or expected
   outputs. Published examples can therefore be stale if labels/text changed. No
   claim is made that the user's remote dataset actually is stale; it was not read
   during this review.
3. Publication finds results by query text rather than case ID. Duplicate questions
   can collide. The current fixed dataset has unique queries, but the mechanism
   does not enforce that constraint.
4. Current run metadata stores a prompt version, but not a prompt hash. The CLI
   creates its `started_at` metadata after running. This review captures metadata
   before running and retains the prompt snapshot/hash separately.
5. Publishing classifier-only runs still attaches handler/route evaluators, although
   that mode does not evaluate handler selection. Use route-mode reports for the
   shown handler/route claims. Published evaluator comparisons do not independently
   incorporate the local result's error/skipped status.

These are review findings, not fixes silently applied during revalidation.

## Demo guidance

Show the saved experiment/report and its failed case. Say “66 of 67 routing cases
matched in this run,” rather than “the router always clarifies ambiguous requests.”
If a live router clarification is needed, “Which is better?” passed the direct
repeats here, but preload a trace because live results can vary. “What is the stock
price?” is a deterministic router clarification example only for the unwrapped
request with market quotes enabled. Avoid it on the Market page until the wrapper
handling is corrected.

## Retained artifacts

- `routes.json`: complete scored run, including the failed case.
- `clarification-repeats.json`: five repetitions of each diagnostic query.
- `market-wrapper-repeats.json`: three repetitions of each wrapper diagnostic.
- `probe.json`: initial successful direct provider check.
- `progress.json`: intermediate/per-case copy for run recovery.
- Sibling `20261031T151105Z/probe.json`: failed sandbox-network probe, excluded
  from the scored run. Network-enabled execution then completed successfully.


## Follow-up: wrapper fix

The generic-query check now extracts the question from the known Market scope
wrapper before phrase matching. Full input is preserved for classification and
specialist execution. Wrapped and plain generic/specific market requests, feature
flags and question-marker edge cases are covered. The router/evaluator/reporting
and greetings suite passed **163 tests** with external guardrail calls and tracing
disabled for offline execution; Ruff passed. The live observations above remain
the pre-fix evidence and were not rerun. The API contract and clarification-policy
findings remain separate.


## Follow-up: API classification error contract restored

The API now maps `ClassificationOutputError` to HTTP 502,
`ClassificationProviderError` to HTTP 503, and `ClassificationTimeoutError` to
HTTP 504 through the shared safe-error handler used by query and resume. Only a
safe `detail` is exposed. The UI client already handles these statuses without
automatic retries. Ordinary domain results and valid clarification responses keep
their existing contracts.

The original 15 failures are resolved. The selected classifier/router/greeting,
UI API-client and clarification-contract suite passed **291 tests**, including
six new typed-error regressions across `/query` and `/query/resume`, with external
guardrails/tracing disabled for offline execution. Ruff passed. Historical review
results above remain unchanged. Clarification policy and publication-tooling
findings are separate and remain open.
