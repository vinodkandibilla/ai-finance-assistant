# Finance Q&A grounding: MVP

A small evaluation helper checks citations, then optionally uses an LLM to review
whole-answer semantic support. It runs on demand, outside the production agent.

## Implemented

- Check response citations against the handbook pages in the evidence actually
  supplied to the answerer. Page metadata must be positive integers.
- Flag unknown/malformed citations, missing citations on answered responses and
  empty answer text. Duplicate citations are checked once.
- Record optional `manually_supported=True/False` and `review_notes`.
- `evaluate_grounding_with_llm` reviews all material claims against cited passages
  using structured verdicts: supported, unsupported or uncertain.
- Judge timeout, provider failure or invalid output returns `needs_review`. Invalid
  citations are rejected before any judge call. Human and LLM results are separate.
- Return `passed`, `failed` or `needs_review`, with issues and invalid citations.

Valid citations alone produce `needs_review`. Pass `True` only after checking
**every material factual claim** against the cited passages, including numbers,
conditions and qualifications. Partial support counts as `False`. A positive
human review cannot override an invalid citation or missing answer text.

For an `insufficient` response, citations are optional but human or LLM review is
still required: check that it abstains without adding unsupported facts. This does not
establish that the original question was truly unanswerable. Other response
statuses remain `needs_review` unless a citation/content check already failed.

## Usage

```python
from finassist.evaluation.finance.grounding import evaluate_grounding

# response: existing AgentResult; evidence: actual retrieved handbook passages
result = evaluate_grounding(response, evidence)  # initially needs_review

# After a human checks the answer against its cited passages:
result = evaluate_grounding(
    response, evidence,
    manually_supported=True,
    review_notes="Reviewed all claims against cited passages; no unsupported facts.",
)
print(result.model_dump_json(indent=2))
```

Only the exact citation format `Financial Education Handbook, page N` is supported.
The helper assumes one trusted handbook corpus. Matching a page number proves
citation presence, not that the passage supports the answer. Supply actual
retrieved evidence, not all gold reference pages. Human review is caller-supplied;
the helper does not independently verify it or enforce reviewer identity.

## LLM judge usage

```python
from finassist.evaluation.finance.grounding import evaluate_grounding_with_llm

# Inside an async function; use the answer and actual retrieved evidence:
result = await evaluate_grounding_with_llm(response, evidence)
print(result.model_dump_json(indent=2))
```

This explicitly makes a billable OpenAI call using `finance_evaluator_model` and
the configured API key. Pass `model=your_chat_model` to override it or inject an
offline model in tests. The default deadline is 10 seconds with zero provider
retries. No calls are made merely by importing the module.

The judge receives answer text, response status and **only cited retrieved
passages**. Its instructions require support for every material claim, including
numbers and qualifications; partial support fails. For an insufficient response,
it checks clean abstention rather than whether an answer exists elsewhere.
`judge_supported` and `judge_reason` record its decision; `manually_supported`
remains unset. Only a supported verdict passes; uncertainty/failure needs review.

An LLM verdict is an automated assessment, not proof of factual correctness.
Calibrate against a few human-reviewed examples before relying on its scores.
No live judge baseline or calibration has been run yet.

## Not implemented / future work

- Claim extraction, per-claim citation association and coverage scores.
- Inline citation parsing, multi-document provenance and conflicting-source checks.
- Retrieval quality, answer completeness, numerical verification or live baselines.
- Batch orchestration, aggregate scores, thresholds or CI integration.

Add these only when observed failures justify them. The earlier claim-level
framework and judge interface were removed to keep the MVP small. The manual function
is synchronous; the separate LLM entry point is async. Both take response and
evidence directly.

## Tests

`cases.jsonl` contains ten synthetic check/review scenarios, not real answers from
the 25 retrieval questions. Each records inputs, expected status and invalid count. Additional offline tests
cover judge verdicts, cited-evidence filtering, malformed output, failure and timeout.

```bash
pytest -q tests/test_finance_grounding_evaluation.py
```
