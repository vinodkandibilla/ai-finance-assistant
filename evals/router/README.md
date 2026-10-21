# Router decision-eval cases

`cases.jsonl` is the versioned, human-labeled dataset for router decision
evaluations. Each line is one independent JSON object. The dataset is public test
content only; it must not contain credentials, customer records, or private queries.

## Case schema

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Stable unique case identifier. Do not reuse it for a different query. |
| `query` | yes | User text sent to the router. |
| `expected_classified_intent` | yes | Expected validated LLM intent, or `null` when a greeting bypasses the classifier. |
| `expected_handler` | yes | Handler selected by deterministic routing, or `null` when a greeting ends before routing. |
| `tags` | yes | Reporting labels such as `market`, `news-boundary`, or `clarification`. |
| `critical` | yes | Whether a regression should fail the critical-case score. |
| `notes` | no | Short maintainer rationale; not sent to the model. |

Allowed classified intents are `finance_qa`, `tax`, `market`, `portfolio`,
`goals`, `news`, `clarify`, and `other`. Allowed handlers are the corresponding
`*_node` values, `clarify_node`, and `unsupported_node`.

The expected classification describes the LLM decision. The expected handler
describes the result after `_route_intent` applies deterministic rules. These can
differ for generic market requests when market quotes are enabled. Standalone greetings have
`null` classification and handler because `greetings.py` handles them before the
classifier is called.

Review labels before changing `INTENT_ROUTER_SYSTEM_PROMPT`. Add a new case rather
than changing the meaning of an existing `id`, so score comparisons remain useful.


## Evaluation helpers

`finassist.evaluation.router.evaluation` provides strict case loading and two async helpers:

- `evaluate_classifier(cases, classifier)` calls only the classifier and compares
  its validated intent. Null-labeled greeting cases are recorded as `skipped`.
- `evaluate_routes(cases, classifier)` uses the production `_classify_intent`
  entry point, including greeting bypass and deterministic handler selection.
  Both intent and handler must match to pass. No domain agents are invoked.

Supply the production classifier for actual LLM evaluation (requires configured
OpenAI credentials and makes billable API calls):

```python
import asyncio
from pathlib import Path

from finassist.workflow.classifier import create_intent_classifier
from finassist.evaluation.router.evaluation import (
    evaluate_classifier,
    evaluate_routes,
    load_router_cases,
)


async def main():
    cases = load_router_cases(Path("evals/router/cases.jsonl"))
    classifier = create_intent_classifier()
    classification_results = await evaluate_classifier(cases, classifier)
    route_results = await evaluate_routes(cases, classifier)
    for result in classification_results + route_results:
        print(result.model_dump_json())


asyncio.run(main())
```

Each helper runs cases sequentially. Running both helpers classifies each
non-greeting query twice; their LLM predictions can differ. Pass a stub implementing
`async classify(query)` for offline tests instead.

Results retain the complete case (including ID, labels, tags, and critical flag),
evaluation mode, status (`passed`, `failed`, `skipped`, or `error`), predicted intent
and handler, match flags, and elapsed milliseconds. Handler match flags are null
in classifier-only mode. Match flags remain null when classification fails.
Known classification failures are recorded as `timeout`, `provider_unavailable`,
or `invalid_output` without storing provider exception text, and subsequent cases
continue. Unexpected exceptions propagate so implementation errors are visible.

Route evaluation uses the current application settings without modifying them.
The initial dataset expects `market_quotes_enabled=True`; disabling it makes the
generic market clarification case fail. Greeting results use the absence of
`classified_intent` and `executed_handler` as null, rather than treating the
production greeting's internal `intent="other"` as an LLM classification.

The loader rejects empty datasets, duplicate IDs, unknown fields, invalid field
types or label values, blank IDs/queries/tags, and inconsistent null labels.
Invalid records include their JSONL line number in the error. Aggregate scoring
and command-line execution are outside these helpers.

## Deterministic router decision tests

Run `pytest -q tests/test_router_decisions.py` for offline decision regression tests.
Each dataset case runs independently with its case ID displayed by pytest and its
labeled intent supplied by a mock classifier. Tests call the production router
entry point directly, independently of the evaluation result/scoring helpers.
They verify greeting bypass, intent preservation, and selected handlers; additional
cases cover generic market normalization, feature flags, specific-symbol queries,
and incomplete requests that must stay in their classified domain.

These tests explicitly control market settings and make no LLM or domain-tool
calls. They validate routing given a supplied classification; they do not measure
the LLM's ability to choose that classification.

## Boundary and greeting coverage

The dataset includes paired examples for single-principal compounding versus
monthly savings, retirement-account rules versus retirement savings targets,
portfolio concepts versus personal holdings, and quotes versus earnings news.
Additional cases cover incidental retirement language and UI-wrapped tax queries.
Each example includes an explicit expected intent and handler for live evaluation.

Greeting cases include case/whitespace/punctuation variants, afternoon/evening
salutations, and UI-wrapped greetings. Greetings followed by finance, market, tax,
or unsupported tasks must reach classification, including within a UI wrapper.
`tests/test_greetings.py` also guards against greeting-prefix false positives and
checks the full graph's greeting response and classifier bypass.

The dataset now contains 63 cases. Offline routing tests supply the labeled
intents; live classifier evaluation is needed to measure boundary classification
accuracy.

## Scores and run metadata

Use `create_run_metadata` before evaluating and `build_router_report` afterward:

```python
from finassist.core.config import get_settings
from finassist.evaluation.router.reporting import create_run_metadata, build_router_report

# Inside an async function, after loading cases:
settings = get_settings()
metadata = create_run_metadata(
    cases,
    mode="classifier",
    model=settings.openai_model,
    system_prompt=settings.intent_router_system_prompt,
    router_timeout_seconds=settings.router_timeout_seconds,
    market_quotes_enabled=settings.market_quotes_enabled,
)
results = await evaluate_classifier(cases, create_intent_classifier(settings))
report = build_router_report(results, metadata)
print(report.scores.model_dump_json(indent=2))
for issue in report.issues:
    print(issue.model_dump_json())
Path("/tmp/router-report.json").write_text(report.model_dump_json(indent=2))
```

For route evaluation, use `mode="routes"` and `evaluate_routes`. Supply the actual
model, prompt and settings used in the evaluation, including prompt overrides or
an explicit offline model label when using a stub. Do not change routing settings
during a run. The helpers do not discover configuration from a classifier.

Accuracy is passes divided by evaluated cases; critical accuracy uses only
critical evaluated cases. Skips are excluded, errors count as unsuccessful, and
an empty denominator produces JSON `null`. Scores include evaluated and critical
evaluated denominators plus passed/failed/error/skipped counts. Route passes
require both intent and handler matches. The issues list contains failed/error
case IDs, critical flags, expected/actual labels, and safe error categories.

Metadata records the UTC start timestamp, model, prompt SHA-256, evaluated-case
SHA-256, classifier timeout and market feature setting. The case hash covers a
canonical JSON representation of the selected cases in order, including labels;
it is independent of JSONL whitespace. Different subsets yield different hashes.
Reports reject mixed evaluation modes, duplicate case IDs and a dataset selection
that differs from the captured metadata. Reports also retain detailed results.
Credentials and raw prompt text are not included in metadata; reports contain
dataset query text through the detailed results.

This reporting is specific to router evaluation runs. It does not execute live
calls automatically, monitor application traffic, enforce thresholds, or generate
a confusion matrix.

## Prompt comparison workflow and acceptance rules

1. Run the offline suite from the repository root:

   ```bash
   pytest -q tests/test_router_*.py tests/test_classifier_*.py tests/test_greetings.py
   ```

2. Before editing the prompt, run the live helpers above and save separate baseline
   reports for classifier and route modes. Capture metadata before each run. Keep
   the baseline prompt text/version alongside the report; its hash cannot restore
   the prompt. Record the source commit as well.
3. Change only the prompt. Use `create_intent_classifier(settings,
   system_prompt=candidate_prompt)` and pass that same text to
   `create_run_metadata(system_prompt=candidate_prompt, ...)`. Save candidate
   reports separately from baselines.
4. Compare runs within the same mode. Model, dataset hash/selection/order, timeout,
   and market feature setting must match; the prompt hash is the intended change.
   Dataset or configuration changes require a new baseline. Inspect per-case
   issues as well as scores, especially boundary cases and greetings.
5. If results vary or the score difference is small, repeat both variants with
   the same number of runs. Retain every report, including failures; a single
   improved run is not sufficient evidence of a consistent improvement.

Acceptance rules for review (manual; reporting does not enforce them):

| Check | Acceptance rule |
| --- | --- |
| Deterministic tests | All pass. |
| Critical cases | No previously passing critical case becomes failed or errored; critical accuracy must not decrease. |
| Overall accuracy | Candidate must meet or exceed baseline in each evaluated mode. |
| Evaluation errors | Resolve errors and rerun affected comparisons before accepting a prompt; retain original error reports. Errors stay in the score denominator. |
| Skips and coverage | Same case IDs and skip behavior across variants; null accuracy is insufficient evidence. Greetings skip only in classifier mode. |

These are regression rules, not a claim that the baseline is sufficiently accurate.
Set an absolute live-accuracy target after reviewing the first baseline and its
remaining failures; no fixed percentage has been established yet. Document any
approved exception and the affected case IDs before accepting the change.

Live evaluation is on demand and incurs provider charges. The optional CLI remains
pending. CI integration is deferred: first check whether existing CI runs the full
pytest suite, then add deterministic coverage only if needed. No live calls should
be added to routine tests or CI as part of this workflow.


## Compact router prompt validation

The router prompt was reduced from 545 to 279 words. See the
[retained candidate comparison](comparisons/20261021T055154121441Z/README.md).
It matches saved baseline classifier/route accuracy and critical-case results in
one run per mode. The first candidate's regression is also retained in the sibling
comparison directory. Exact model input-token savings and latency are not measured
by word counts. Restart the API to load the prompt; an explicit environment prompt
override takes precedence over the source default.
