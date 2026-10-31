# Cost and latency controls

FinPilot bounds model/tool work and reuses eligible results. These are implemented controls, not a measured production cost guarantee. Use [evaluation and observability](evaluation.md#observability) to establish a baseline before tuning them.

| Control | Current behavior |
| --- | --- |
| Models | `gpt-4o-mini` for routing and most reasoning; `text-embedding-3-small` for dense embeddings |
| Finance/Tax answer caches | Separate Qdrant caches for eligible public definitions, default TTL one hour; cache hits still pass initial guards and routing |
| RAG budgets | Three retrieval actions total, 15 passages and 30,000 characters; Finance baseline three passages, Tax five |
| Reranking | Optional cross-encoder disabled by default |
| Timeouts | Router 10 seconds, hybrid retrieval 15 seconds, most specialists 60 seconds, Market 30 seconds overall |
| Market cache | Process-local TTL/LRU; quotes 30 minutes, symbol/history results 24 hours, identical in-flight calls shared |
| Market retries | Eligible transient failures only, two retries after the initial attempt; per-MCP-attempt timeout and jitter remain deferred |
| News acquisition | Up to three searches, stopping early when evidence is sufficient |

Persistent document vectors avoid re-embedding the whole corpus on every restart. Each search can still generate query embeddings; a changed query is not automatically served by a general query-embedding cache. No persistent News response cache is implemented.

Compare representative queries using model calls, input/output tokens, provider-reported cached tokens, latency and recorded cost. Semantic answer caching and OpenAI prompt caching are different mechanisms. The OpenAI usage/billing dashboard is the billing authority; request summaries may omit nested usage. Saved-prediction LangSmith experiment latency is replay latency, not live inference latency.

Future work includes evaluation-driven model selection, embedding reuse, prompt compression, explicit per-agent spend budgets and production-wide provider limits. Model changes and shorter prompts need regression evaluation, not just a cost estimate. Project-level spend alerts are configured in the provider dashboard.

Configuration sources: [application](../src/finassist/core/config.py), [Finance](../src/finassist/agents/finance_qa/config.py), [Tax](../src/finassist/agents/tax_education/config.py). See [API operations](api.md) for environment setup.
