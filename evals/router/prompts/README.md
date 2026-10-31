# Router prompt variants

- **A — baseline.txt:** retained baseline prompt snapshot.
- **B — candidate.txt:** retained candidate prompt. It matches the earlier production-default `v2` text after ignoring trailing
  whitespace. The current `v3` adds customer-profile routing; A/B snapshots remain
  unchanged so saved experiments keep their original meaning.

`scripts/evaluate_router_langsmith.py` runs either snapshot through the same
`with_structured_output(IntentClassification)` classifier and can publish saved
predictions as a LangSmith experiment with `--publish`. Candidate B explicitly requests JSON; baseline A requests the required structured
intent. Supplying a variant does not change production settings.

Use the same model, dataset selection, settings and mode for comparisons. Review
case failures, not just aggregate scores. Publication replays saved predictions,
so the dashboard target latency is not live inference latency. See the
[latest revalidation](../revalidations/20261031T151125Z/README.md) for results and
known evaluation-tooling limitations. Do not treat a single run as proof of stable
clarification behavior.
