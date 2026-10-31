# Architecture sequence review

Historical review record. For current behavior, use the linked execution sequences, [API operations](../api.md) and [latest evaluation findings](../evaluation.md).

Review date: October 27, 2026. Scope: source-grounded documentation in this directory. The implementation contains uncommitted changes; this describes the inspected working tree, not necessarily EC2 or an earlier installed wheel.

## 1. Review method

1. Trace UI entry points through HTTP adapters, FastAPI validation/dependencies, parent graph, specialist construction, nested graphs/tools, output checks, and UI rendering.
2. Inspect alternate branches rather than only the happy path: cache hit/miss/bypass, greeting/unsupported/blocked input, missing provider key, evidence gap, calculation-only request, interrupt/resume, and malformed completed clarification.
3. Follow store/client construction to establish actual ownership, persistence, and provider boundaries. Search for cache implementations instead of labeling every persistent file a cache.
4. Compare source defaults with written numeric limits. Distinguish model instructions from Python validation, and requested retrieval depth from underlying effective caps.
5. Verify document-local links and absence of diagrams. Review all six specialist sections for providers, tools, data stores, caches, human interaction, validation, completion, and failure limits.
6. Do not call external paid services or claim a live application end-to-end test as part of this documentation task. Documentation checks cannot prove live model correctness or production readiness.

## 2. Review findings incorporated into the documents

1. Portfolio uses seeded in-memory SQLite. It is not the file-backed clarification database and does not obtain current quotes from Market.
2. The checkpoint path's Finance-oriented name is misleading: the parent graph checkpointer supports clarification across specialists.
3. Graph construction can initialize retrieval indexes on first dependency use. The current system does not have a separate always-running ingestion/API indexing service.
4. Tax's optional query analyzer exists but is bypassed by the active explicit-input recovery path. Multipart decomposition is prompted through model tool arguments, not guaranteed by a planner node.
5. Default Finance/Tax reranking is disabled. RRF evidence ordering and optional cross-encoder reranking must be represented as separate branches.
6. A configured Tax expansion limit of 40 does not override the underlying shared hybrid retriever candidate cap. The document avoids claiming an unconditional 40-passage expansion result.
7. Finance explicitly validates its dense-query list limit in the active path. Tax has a prompt/tool-interface expectation but not identical active recovery validation; this is not presented as equivalent enforcement.
8. A semantic answer-cache hit skips specialist work, not initial request safety checks and intent routing. Cacheability is restricted by concept, namespace, expiry, supported citations, and result shape, not similarity alone.
9. Market quote/history cache, tool-catalog cache, and session reuse are three separate mechanisms. They are process-local, not an API gateway cache or distributed rate limiter.
10. No general application disk-backed HTTP/news cache was found. Persistent Qdrant, local model files, SQLite checkpoints, and UI retained results are named separately.
11. Only Market uses MCP. News uses direct Tavily HTTP; Finance/Tax use Qdrant clients and OpenAI embeddings; Goals/Portfolio use deterministic local tools.
12. A normal generic-router clarification is not the checkpointed specialist human-in-the-loop flow. Only real interrupts support `/query/resume`.
13. Valid Market clarification can have an empty parent answer at interruption time. API finalization now preserves that paused response; a graph child interrupt is not automatically a crash.
14. API-boundary and graph output redaction are distinct checks. Only answer text is routinely sanitized; structured calculations/citations are not recursively protected by that node.
15. Root trace previews are redacted and narrowed, but this does not establish complete trace-data sanitization across all child messages and public provider payloads.
16. News UI publication period is translated into text and model-selected tool parameters. Strict filtering of returned publication dates is not fully enforced; Today does not currently guarantee all cards were published today.
17. Portfolio dashboard operation remains model-selected despite clearer role-specific prompts. Authorized manager context does not itself force correct routing or `screen=all`.
18. Lowest/highest-price calculations have a 100-calendar-day window; provider request output caps are counts of records. They are not the same unit.
19. Request usage summaries are partial, not comprehensive token/cost attribution. Runtime evidence assessment and offline evaluation are separate flows.
20. Single-server testing does not establish autoscaling, durable UI history, real authentication, global provider quotas, or production deployment readiness.

## 3. Evidence boundaries and unresolved implementation risks

1. The architecture was checked statically against implementation paths. Some supporting tests elsewhere in the project have known environment/provider failures. This document does not label those as passing or treat historical test counts as current verification.
2. Native dependency compatibility has previously prevented portfolio table rendering and spaCy loading. Regex fallback is explicit, but full NLP availability still depends on environment repair.
3. PII can still mistake financial values for phone/account identifiers. Driver-license detection was removed; no general verified-value exemption mechanism was implemented.
4. The competitor regex still blocks names such as Robinhood, including legitimate research questions. It is separate from domain classification and remains part of the current behavior.
5. Model-driven operation selection, scope interpretation, decomposition, and evidence sufficiency remain probabilistic. Code budgets and tool access checks constrain execution but do not guarantee a correct final answer.
6. Resume checks compare simulated context, and full initial-guard parity is deferred. Production identity/session isolation needs a separate design.
7. Local Qdrant paths depend on package placement; multiple processes/replicas and file-backed SQLite require explicit deployment design. Do not draw a horizontally scalable shared store by default.
8. Reranker/evaluator/analyzer source files are not proof those components run in the active path. The documents identify optional or bypassed paths explicitly.

## 4. Acceptance checklist for diagram authors

1. Can every runtime box be mapped to a linked implementation file? Yes, source anchors accompany each major sequence.
2. Are all six agents covered without claiming every request invokes all agents? Yes.
3. Are UI-local calculations and session state separate from backend reasoning and persistence? Yes.
4. Are actual data providers, MCP subprocess, embeddings, Qdrant collections, portfolio SQLite, and checkpoint SQLite distinct? Yes.
5. Are cache lookup order, invalidation/expiry, hit/miss behavior, and storage scope explicit? Yes.
6. Are human clarification, blocked input, insufficient evidence, execution errors, and successful completion separate? Yes.
7. Are architecture aspirations and known gaps distinguishable from current execution? Yes.
8. Are documents sequences rather than diagrams? Yes; numbered steps are provided for later diagram construction.
