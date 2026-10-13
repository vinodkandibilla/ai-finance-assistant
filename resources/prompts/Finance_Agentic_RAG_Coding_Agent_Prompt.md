# Coding Agent Prompt: Investment-Education Finance Q&A Agentic RAG

## 1. Role and objective
Act as a senior Python/LangGraph engineer. Implement a runnable, modular, tested **investment-education-only** Finance Q&A Agentic RAG node that plugs into an existing multi-agent router. Deliver real Python 3.10+ code, not pseudocode. Optimize for correctness, controlled latency/cost, explainability, and maintainability. Do not silently implement additional agents.

## 2. Strict domain boundaries and routing
**In scope:** stocks, bonds, ETFs, mutual funds, index funds, benchmarks (Nasdaq-100 vs Nasdaq Composite), market terminology (P/E, EPS, NAV, market cap, dividend yield, volume, 52-week range, borrow rates, short inventory), options basics (calls, puts, Delta, Gamma, Theta, Vega, Rho, moneyness, intrinsic/extrinsic value, assignment, contract multipliers), and investing risk concepts (diversification, allocation, volatility, concentration, rebalancing, risk tolerance versus capacity).

**Out of scope:** all retirement-account topics (Roth IRA, Traditional IRA, 401(k), SEP/SIMPLE and other retirement plans), their advantages and comparisons; all tax education, tax treatment, filing, IRS forms, deductions, credits, immigration and residency; live stock prices, breaking financial news, personalized portfolio analysis, individual financial planning, trades, and account actions. These belong to dedicated agents. Do not index out-of-scope documents, add examples of them, or answer them here. An out-of-scope question must return a structured handoff result to the top-level router with a destination intent and a concise reason; do not fabricate a destination implementation. Mixed-intent requests must return all detected intents for top-level orchestration. Keep this boundary enforceable with tests.

## 3. Required workflow
Top-level router -> Finance Q&A entry -> scope and query-completeness analysis -> clarification if essential -> deterministic financial metadata extraction -> structured query decomposition -> concurrent dense and sparse retrieval -> RRF -> cross-encoder reranking -> context sufficiency evaluation -> [answer OR one targeted retry] -> grounded answer. A new message received during clarification must go through the top-level router again, not blindly resume the previous intent.

## 4. Query completeness and clarification
In the same structured query-analysis LLM call, decide whether the question is sufficiently specified to search. Examples: 'What is Delta?' is complete; 'Which is better?' without comparands is not. Output `needs_clarification`, `clarification_question`, `detected_intents`, `is_complex`, `sparse_keywords`, and 1-3 `dense_sub_queries`. If clarification is needed, short-circuit retrieval. Show how the parent LangGraph can use `interrupt()` with persistent checkpointer/thread_id and `Command(resume=...)`; on resume re-run the top-level router to catch new or multiple intents. If integrating with the existing parent graph is impossible, provide an explicit adapter interface rather than pretending the child graph owns the router.

## 5. Deterministic financial metadata extraction
Do not call an LLM to infer Qdrant filters. Load the **actual financial corpus** metadata vocabulary/index schema at startup; validate against it. Possible metadata only if the corpus has it: `topic`, `subtopic`, `asset_class`, `instrument_type`, `ticker`, `document_type`, `source`, `as_of`. Recognize explicit tickers, asset terms, and known topics conservatively; ambiguous strings (e.g., common English words that resemble tickers) must not become hard filters. Never copy metadata from a tax/retirement corpus. Only apply hard filters when confidence is high and the indexed metadata supports them; otherwise search without optional filters. Separate freshness metadata from live-market-data functionality.

## 6. Query decomposition and lexical terms
Use LangChain `.with_structured_output(PydanticModel)` with a configurable inexpensive LLM. Preserve the original question. For simple queries, one dense subquery is sufficient. For complex comparisons, create 1-3 independently meaningful dense subqueries and extract exact lexical terms for BM25 (e.g., `QQQ`, `Nasdaq-100`, `Delta`, `Theta`). Do not force decomposition when it adds no value. If analysis fails, fall back to the original question as one dense query and derive conservative lexical terms deterministically.

## 7. Financial corpus and ingestion assumptions
Use the repository's retained financial RAG source as the corpus: `src/ai_finance_assistant/rag/finance-qa/Financial_Education_Handbook_integrated.pdf`. The PDF and companion DOCX are retained assets; no finance loader, index, or graph route is currently active. If implementing the finance agent described by this prompt, use the PDF as its authoritative corpus and do not silently substitute the DOCX, tax JSONL, external web content, or another handbook. Preserve its original source path/title in citations. The prior finance loader (`src/ai_finance_assistant/rag/finance_education.py`) has been removed; implement and test a finance-specific ingestion adapter as part of this work. The former chunker targeted 800 characters with 120-character overlap and attached `document_title`, `jurisdiction`, `chapter`, `section`, `page_number`, and `chunk_id`; verify these assumptions against the PDF and current project before reusing them. Build the dense and BM25 indexes from the same PDF-derived chunks and version, and document how both indexes are refreshed together. Keep the investment-education scope in this prompt authoritative: if the handbook contains material outside that scope, do not use it to answer those requests. Do not assume IRS JSONL files, tax documents, or immigration data are available.

## 8. Dense retrieval
Use Qdrant with an injectable embedding client. Execute a separate dense retrieval for each decomposed query; when there are three subqueries and `DENSE_TOP_K=3`, there are up to nine dense result positions (not necessarily nine unique documents). Run independent searches concurrently with bounded concurrency, timeout, and optional validated metadata filters. Normalize results to typed objects containing doc_id, text, metadata, source, score, and rank.

## 9. Sparse retrieval
Use `rank_bm25` over the same document corpus. For the MVP, combine extracted keywords into one lexical query and retrieve one ranked sparse list; no need for one BM25 call per dense subquery. BM25 is lexical ranking, not strict exact-match-only filtering. Document tokenization/normalization of ticker symbols, hyphens, acronyms, and options terms. Use a thread executor if CPU-bound BM25 ranking would block the event loop.

## 10. Concurrent hybrid retrieval
Run dense-search tasks and the sparse-search task concurrently with `asyncio.gather` or `TaskGroup` where appropriate; do not claim concurrency for synchronous blocking calls. If one retriever fails, degrade to the surviving retriever with warning/metrics. If both fail, return a controlled failure. Distinguish missing matches from service failure.

## 11. Reciprocal Rank Fusion (RRF)
Fuse **each** ranked list (three dense lists plus one sparse list in the three-subquery example) using `RRF(d) = sum(1 / (60 + rank_i(d)))`, with ranks starting at 1. Keep duplicate doc IDs across input lists so they contribute multiple times; deduplicate in the *fused output*, not before score accumulation. Do not combine incomparable raw BM25 and vector similarity scores. Sort by fused score and cap at `RRF_CANDIDATES=20`, or fewer if fewer unique candidates exist.

## 12. Cross-encoder reranking
Use `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` with injectable interface and configurable batch size. Score `[original_user_query, candidate.content]` for up to 20 RRF candidates. Sort descending and retain `RERANK_TOP_N=5` by default. The reranker produces relative relevance scores, not calibrated probabilities. Preserve doc IDs, provenance, and citations. For a comparative query, avoid dropping evidence necessary for one side of the comparison; evaluate coverage downstream.

## 13. Context sufficiency evaluator
Use a **separate structured LLM evaluation** of the original query against the selected *set* of chunks. Return `sufficient: bool`, `sufficiency_score: float` (0-1, heuristic and NOT a calibrated probability), `missing_concepts: list[str]`, `suggested_queries: list[str]`, and `reason: str`. Distinguish relevance from completeness: relevant Delta chunks alone may be insufficient for a Delta-and-Theta question. Require grounding in actual retrieved text. Avoid assuming a universal fixed cutoff like 0.8; define a configurable policy, calibrate against an eval set, and log reasons.

## 14. Single self-correction retry and best-context tracking
Set `MAX_RETRIEVAL_RETRIES=1`: initial attempt plus **at most one** targeted retry. On insufficient context, use evaluator-suggested missing concepts to reformulate, retrieve, fuse, rerank, and evaluate once more. Save both attempt contexts, scores, and missing-concept coverage. Choose the **best-supported** context, not automatically the latest: if attempt 1 scores 0.80 and attempt 2 scores 0.60, retain attempt 1. Scores are heuristic; break close ties with explicit coverage/evidence checks and avoid treating tiny score differences as statistically meaningful. Optionally merge complementary evidence only if it remains source-grounded and fits the context budget. Never loop indefinitely.

## 15. Final generation
Use the original user question and selected evidence with source identifiers. Answer only supported investment-education concepts; cite retrieved chunks. Clearly state when evidence is insufficient and do not invent specifics. No personalized investment recommendations, transactions, live prices, tax/retirement explanations, or agent handoffs disguised as answers. Return structured `answer`, `citations`, `status`, and optional `handoff_intents`/`clarification_question` to the parent application.

## 16. LangGraph state and conditional routing
Define typed state (TypedDict or Pydantic) for original query, scope/intents, clarification, metadata filters, dense queries, sparse keywords, per-source ranked lists, fused candidates, reranked context, evaluator result, retry_count, best_context, best_score, final response, errors and timings. Nodes: scope_and_analyze, extract_metadata, hybrid_retrieve, fuse_rrf, rerank, evaluate, reformulate, generate, handoff, clarify. Conditional edges: out-of-scope -> handoff; incomplete -> clarify; sufficient -> generate; insufficient with retry remaining -> reformulate; exhausted -> generate limited answer. Clearly show parent-router integration and interrupt/resume adapter.

## 17. Configuration, performance, and reliability
Externalize `DENSE_TOP_K=10`, `SPARSE_TOP_K=10`, `RRF_K=60`, `RRF_CANDIDATES=20`, `RERANK_TOP_N=5`, `MAX_RETRIEVAL_RETRIES=1`, model names, timeout, max parallelism, and cost budget. Do not hardcode secrets. Cache embedding/query analysis only when appropriate, and avoid caching user-specific sensitive text indiscriminately. Provide explicit error handling for malformed LLM output, embedding failures, Qdrant unavailability, sparse-index mismatch, reranker failure, empty retrieval, and timeouts. Avoid using a 0.8/0.9 vector score cutoff without calibration.

## 18. Evaluation and observability
Instrument per-stage latency, model calls, tokens/cost estimates, dense/sparse hit counts, deduplicated candidate count, rerank count, sufficiency outcome, retry count, winning attempt, citation coverage, and handoffs. Avoid logging secrets or sensitive user data. Provide an offline evaluation harness for retrieval recall@k, MRR/nDCG, citation correctness, query-decomposition quality, scope-routing accuracy, sufficiency calibration, and retry benefit. Report baseline single dense search versus agentic hybrid pipeline, including latency/cost tradeoffs.

## 19. Required tests (fully mocked, no live API credentials)
- Options: 'How does Delta change for an in-the-money call as expiration approaches, and how does Theta affect it?' Verify two concepts are covered and missing Theta triggers retry.
- ETF vs benchmark: 'What is the difference between buying QQQ and tracking the Nasdaq-100?' Verify ticker and index lexical preservation, comparison evidence, and citations.
- Financial basics: 'Explain NAV versus market price for an ETF.' Verify simple-query path and no unnecessary decomposition.
- RRF: 3 dense top-3 lists plus 1 sparse top-3 list, repeated doc IDs, 1-based ranks, correct accumulated scores, deduplicated output.
- Retry: first sufficiency 0.80, second 0.60 -> first context wins; first 0.70, second 0.81 -> second wins subject to coverage checks; never more than two retrieval attempts.
- No optional metadata -> retrieval still runs; ambiguous ticker -> no hard filter.
- Clarification: 'Which is better?' -> ask clarification before retrieval; new response with second intent -> return to top-level router.
- Out-of-scope routing: 'Compare Roth IRA and Traditional IRA'; 'Explain 401(k)'; 'How do I file taxes?'; 'H-4 visa rules?' -> handoff, no Finance Q&A answer. These strings may appear **only in routing exclusion tests**, never in the finance knowledge base or retrieval prompt.
- Failure modes: dense-only fallback, sparse-only fallback, both fail, malformed evaluator output, empty results.

## 20. Deliverables and acceptance criteria
Produce complete runnable files:
```
finance_qa_agent/
  config.py
  schemas.py
  scope_router_adapter.py
  metadata_extractor.py
  query_analyzer.py
  corpus_index.py
  hybrid_retriever.py
  rrf.py
  reranker.py
  evaluator.py
  generator.py
  agent_graph.py
  requirements.txt
  README.md
  tests/
    test_scope_and_clarification.py
    test_metadata.py
    test_rrf.py
    test_hybrid_retrieval.py
    test_rerank_and_evaluator.py
    test_graph.py
```
First provide a concise architecture summary and assumptions, then full file contents without placeholders or missing imports. Include install/run/test instructions, example mocked run, example parent-router handoff, and a short list of design tradeoffs. Validate Python 3.10 compatibility, LangGraph conditional edges, Pydantic schemas, asyncio usage, 1-based RRF ranks, bounded retries, and fully mocked tests. Do not claim tests passed unless they were actually run. Do not implement the excluded dedicated agents.
