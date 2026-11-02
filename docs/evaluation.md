# Evaluation and observability

Evaluation checks expected behavior over cases. Tracing shows what happened in one execution. FinPilot has live router evaluation, deterministic regression tests and Finance evaluation scaffolding; these do not establish universal answer correctness.

## What is evaluated

| Area | Current evidence and limits |
| --- | --- |
| Router | 71 labeled cases; baseline/candidate prompts; intent, handler and route scoring. Only two cases expect `clarify`, so aggregate accuracy gives limited clarification coverage. |
| Deterministic routing | Stub-controlled tests verify mappings, wrappers, greetings and feature flags. They do not measure the LLM's ability to select an intent. |
| API and UI contracts | Offline tests cover classification errors, request/resume handling, safe client failures and clarification response preservation. |
| Finance | [Evaluation guide](../evals/finance/README.md) separates deterministic/reporting checks from pending live retrieval and grounding measurements. |

## Run router evaluations

Cases live in [cases.jsonl](../evals/router/cases.jsonl); the [router guide](../evals/router/README.md) defines labels, score denominators and comparison rules. These commands make billable provider calls:

```bash
.venv/bin/python scripts/evaluate_router_langsmith.py --variant A --mode routes --limit 5
.venv/bin/python scripts/evaluate_router_langsmith.py --variant B --mode routes --limit 5
```

The CLI variants A and B are retained prompt snapshots; B is v2 and does not select the current v3 default. Use `create_intent_classifier()` with the evaluation helpers in the router guide to evaluate the active default. The CLI defaults to the first five cases. Use `--all` for the full dataset and `--publish` to create a LangSmith experiment when intentionally requested. For A/B comparisons, keep dataset version, selection/order, model, evaluation mode, settings and repetition count consistent. Inspect critical cases and individual failures, not just aggregate scores. The runner's publication limitations are documented in the retained review below.

Run core router tests without live guardrail or tracing calls:

```bash
MODERATION_ENABLED=false INJECTION_CLASSIFIER_ENABLED=false \
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
.venv/bin/python -m pytest -q tests/test_router_decisions.py \
  tests/test_router_evaluation.py tests/test_router_reporting.py
```

These variables apply only to this test process. They do not change the application's guardrail policy.

## Earlier v2 validation

On **31 October 2026**, the production-default prompt `v2` with `gpt-4o-mini` matched **66/67 routes (98.51%)**, including **47/47 critical cases**, with no evaluation errors. This was one current-prompt run, not a new A/B comparison or a production accuracy estimate.

The remaining mismatch was “How much should I invest?”: expected `clarify`, observed `goals`. Five additional repetitions also returned Goals. “Which is better?” returned `clarify` in five direct repetitions. The exploratory “Can you help me with my finances?” returned no `clarify` results in five repetitions, so it is not a reliable demo. These results do not guarantee future classifications or establish the policy label for every ambiguous question.

Two implementation defects found during that review were subsequently fixed:

- The generic-query check now extracts the question from the known Market wrapper before matching; specific-symbol requests still reach the Market agent.
- Typed classifier failures now return HTTP 502 for invalid output, 503 for provider unavailability and 504 for timeout on query/resume. The previous 15 contract failures were resolved without changing expected HTTP assertions.

The selected classifier/router/greeting, UI API-client and clarification-contract suite passed **291 tests** after those fixes, with external guards/tracing disabled for offline execution. Ruff passed. The live score above remains the original pre-fix observation; no new post-fix live accuracy is claimed. The [full review and retained results](../evals/router/revalidations/20261031T151125Z/README.md) preserve run configuration, all observed failures and later fixes. Older reports are historical evidence, not current test counts.

## Observability

LangSmith records the request tree, router and specialist spans, model/tool calls, retrieval stages, cache lookup/storage and available usage metadata. Input guard subchecks are not all separately instrumented; do not claim named moderation or injection timing spans exist. Request summaries are partial, and provider billing remains the charge authority.

Root previews use redacted text. Environment, source and router prompt version can be attached through trace settings. Child tools/messages need their own data-handling review; redacting the parent preview does not establish complete trace sanitization. See [guardrail boundaries](architecture.md#guardrails).

Published router experiments replay saved predictions. Their dashboard target latency and tokens are not measurements of the original inference call; local results retain `elapsed_ms`. A selected request trace can show actual model/tool latency for that run. Neither a single trace nor an aggregate correctness score proves every final answer is correct.

### Request latency percentiles

Run this read-only report using the API key in the repository `.env` or your process environment:

```bash
.venv/bin/python scripts/langsmith_latency.py
.venv/bin/python scripts/langsmith_latency.py --app finpilot-ai-aws --days 30
```

The `--app` option selects the LangSmith project (`--project` remains an alias).
The default project is `finpilot-ai-dev` and the window is seven days. The report
retrieves completed root traces and prints P50, P80, P85, P90, P95 and P99 in
seconds, including completed error traces. It excludes incomplete traces and child
spans, reports the sample size and uses linear interpolation. This measures server
trace duration rather than browser latency. It does not create traces or experiments.
Use the appropriate regional `LANGSMITH_ENDPOINT` and workspace configuration if required.

## Prompt versioning

The source default and `INTENT_ROUTER_PROMPT_VERSION` live in [system prompts](../src/finassist/prompts/system_prompts.py). An explicit environment prompt override takes precedence; restart the backend when changing cached settings. A version label alone does not verify the prompt loaded in an existing process.

The [A/B snapshots](../evals/router/prompts/README.md) are explicit evaluation inputs, not deployment switches. Retain actual prompt text/hash, model, dataset selection, mode and settings with comparisons. Current metadata stores a prompt version but no prompt hash; the revalidation retains its snapshot/hash separately. Broader answer-quality evaluation remains incomplete.


### Customer-profile routing revision

Router prompt `v3` explicitly includes stored customer investment profiles and
retrieval by customer name/ID. Four new cases cover the two reported C011 queries,
general risk-profile education and an unrelated social profile. The dataset now
contains 71 cases; historical 67-case scores above remain unchanged. Portfolio
tools still enforce application identity scope after routing.


The [v3 routing validation](../evals/router/revalidations/20261031T223728Z-profile-v3/README.md)
matched **70/71 routes (98.59%)**, with **51/51 critical cases** and no errors.
Both reported C011 requests routed to Portfolio in the benchmark and in three
additional repetitions each. The earlier v2 prompt reproduced both `other`
misclassifications. General risk-profile education and an unrelated social profile
retained their correct domains. The known investment-amount clarification mismatch
remains; no expected labels were changed. Offline checks passed 163 router-related
and 43 Portfolio routing/tool/authorization tests. These are routing checks, not
proof of permission to view C011 or a live end-to-end profile response.
