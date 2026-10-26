# Router prompt variants

- **A — baseline.txt:** snapshot of `INTENT_ROUTER_SYSTEM_PROMPT` when these variants were created.
- **B — candidate.txt:** user-supplied candidate; wording retained unchanged.

These files do not change the production prompt. Supply their contents through
`create_intent_classifier(system_prompt=...)` for on-demand comparisons, using the
same dataset, model, settings and evaluation mode. The proposed LangSmith runner is
not implemented yet; no live evaluation or upload was performed.

The classifier uses `with_structured_output(IntentClassification)` with an `intent`
field. Candidate B asks for a plain intent string; preserve the same structured
output mechanism for both runs, and review any resulting invalid-output errors.
Do not change the parser for only one variant. Before comparing prompt logic alone,
consider aligning B's output instruction with the existing schema as a separately
recorded candidate revision.
