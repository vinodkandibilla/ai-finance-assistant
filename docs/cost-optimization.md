# Cost optimization

Current controls include `gpt-4o-mini` for most reasoning, answer caching for Finance/Tax, bounded retrieval and evidence expansion, reranker avoidance for the MVP, query limits, and bounded retries.

Use LangSmith to establish a baseline for representative requests: model calls, input/output tokens, latency, cache status, and cost. Project spend alerts and limits belong in the OpenAI project dashboard. Future improvements include prompt compression, model routing, embedding reuse, and per-agent budgets.
