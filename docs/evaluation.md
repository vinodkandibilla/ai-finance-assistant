# Evaluation

Router evaluation cases live in `evals/router/cases.jsonl`. Baseline and candidate prompts are in `evals/router/prompts/`. Evaluators score intent, handler, and route correctness.

Run a small local experiment:

```bash
.venv/bin/python scripts/evaluate_router_langsmith.py --variant A --mode routes --limit 5
.venv/bin/python scripts/evaluate_router_langsmith.py --variant B --mode routes --limit 5
```

Use `--publish` and a configured LangSmith key to publish datasets and experiments. Use `--all` only when a full run is intentional. See the script and [prompt versioning](prompt-versioning.md).
