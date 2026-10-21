# Finance Q&A evaluation status

## What is available

| Area | Implementation | What has actually been measured |
| --- | --- | --- |
| Retrieval | 25 labeled questions (22 answerable, 3 unsupported), reference chunks/hashes and coverage map | Dataset integrity and mocked retrieval-tool behavior only; retrieval relevance has not been scored |
| Grounding | Deterministic citation checks plus optional manual review or on-demand LLM judge | Synthetic offline fixtures and stub judge tests; no live answer-grounding baseline or judge calibration |
| Calculations | Existing calculator tests plus 8 labeled execution cases | Real arithmetic/tool execution with scripted tool choices; not live LLM tool selection or extraction |
| Clarification/scope | Existing agent tests and focused regressions | Scripted decisions, pause/resume and scope-result handling; not live recognition accuracy |
| Failures | Existing runtime/tool tests and focused regressions | Injected empty results, exceptions and deadline cancellation; not provider reliability |
| Reporting | On-demand offline pytest aggregation | Test statuses and artifacts; live areas explicitly marked `not_run` |

## Run and inspect

```bash
.venv/bin/python scripts/evaluate_finance.py
```

Open `evals/finance/reports/<UTC-timestamp>/summary.md` for the overview and
`results.json` for individual test outcomes and metadata. `pytest.xml` holds
assertion details; `pytest.log` includes warnings/collection errors. Each run is
saved separately. This command runs offline checks, not the 25 user questions
through the actual RAG pipeline. A passed area means its selected suites passed,
not that live accuracy is 100%. Shared tests are counted once in the unique totals
and may appear in multiple area totals; some shared suites also test tax behavior.

The LLM grounding helper is available separately:

```python
# In an async function, with an existing answer and its actual retrieved evidence:
result = await evaluate_grounding_with_llm(response, evidence)
```

Import it from `finassist.evaluation.finance.grounding`. This explicit call uses
OpenAI unless a model is injected; it incurs API cost and is not invoked by the
report command. It evaluates support, not retrieval quality or answer completeness.

## Pending / deferred for MVP

- Actual retrieval runner, Hit@k/concept-recall/MRR scores and retrieval baseline.
- Capture actual generated answers/evidence and run/calibrate the grounding judge.
- Live tool-choice/input-extraction checks (manual smoke checks suffice initially).
- Combining real live per-case results/model configuration into the report.
- CI integration and absolute quality thresholds.

No mandatory live target is established by these offline passes. Before changing
retrieval or prompts, capture comparable live evidence if assessing answer quality.
The router baseline is separate and does not establish Finance Q&A quality.

Details: [retrieval](retrieval/README.md), [grounding](grounding/README.md),
[calculations](calculations/README.md), [clarification/scope](clarification_scope/README.md),
[failures](failures/README.md), [reports](reports/README.md).
