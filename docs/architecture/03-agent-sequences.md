# Specialist architecture: ordered execution specifications

Read after the [system](01-system-sequence.md) and [router](02-router-sequence.md) sequences. Each section describes an independent selected specialist. A response does not necessarily visit all six. Shared output checks and API serialization follow each completed specialist; checkpointed clarification can suspend any domain before completion.

## 1. Shared Finance/Tax educational runtime

1. The parent handler first invokes the domain's optional semantic answer cache. Eligibility, exact lookup, semantic lookup, expiry, namespace, and write rules are detailed in the system sequence. A hit returns the supported public outcome and preserves the current canonical question; a miss enters `AutonomousEducationAgent`.
2. Start a domain child graph and an active-execution deadline. Construct model messages from the canonical `query`; do not expect separate `user_query`/`search_query` state channels.
3. Build the system prompt from domain policy plus search budget. Register initial retrieval, evidence expansion, targeted search, clarification, final submission, and domain-specific calculators. A ReAct-style model/tool loop selects calls; tool exceptions are converted into tool-level errors where configured.
4. Baseline retrieval acquires evidence once. Expansion reuses the last plan at greater depth; targeted search changes terms for an explicitly identified evidence gap. These are different actions, not arbitrary repeated copies of the same query.
5. Read prior tool history to track calls, plans, seen evidence, and character usage. Validate action limits, deduplicate passages, select/re-rank evidence, apply passage/character budgets, and return evidence IDs/citations plus remaining-action feedback to the model.
6. The model judges support against the actual requested facts and chooses to submit, clarify user inputs, or recover evidence. This runtime does not require the legacy standalone LLM evaluator/analyzer on each retrieval. Prompt-driven sufficiency is not an independent correctness proof.
7. Essential user detail is obtained through a clarification interrupt. Ordinary clarification tools reject guidance that merely asks for a new unrelated question. A changed domain during resume should finish out of scope rather than dispatch to another specialist.
8. Final completion uses `submit_finance_answer` or `submit_tax_answer` alone with schema-valid fields. If completion is missing/invalid, the runtime allows a bounded corrective attempt. It also allows one recovery turn when an in-scope answer skipped evidence acquisition.
9. Collect retrieval evidence/errors from tool artifacts. Domain calculation processors validate successful calculator results and determine whether a pure-calculation answer has trustworthy numeric support without RAG.
10. An answered explanation without evidence becomes insufficient or error. Cited answers are restricted to supplied sources; a blank answered result becomes an answer-generation error. Completed education results cannot carry a stray resumable clarification question. Blank insufficient results receive an explanatory fallback.
11. Return answer, citations, status, errors, optional calculation data, `final_result`, and domain intent. Catch bounded time/recursion/execution failures with safe messages; rethrow `GraphInterrupt` for checkpoint handling.
12. Default Finance/Tax active time is 60 seconds, search-action budget three, recursion limit 30, one expansion, two targeted searches, at most 15 evidence passages and 30,000 evidence characters. These limits count different units; do not label all of them “retries.” Finance additionally validates at most three dense queries; Tax's prompt requests at most three but its active recovery function does not independently enforce that same list-length setting.

Source anchors: [shared runtime](../../src/finassist/agents/common/runtime.py), [evidence accounting](../../src/finassist/agents/common/evidence_recovery.py), [Finance settings](../../src/finassist/agents/finance_qa/config.py), [Tax settings](../../src/finassist/agents/tax_education/config.py).

## 2. Finance Q&A: ingestion and retrieval

1. Resolve the packaged integrated Financial Education Handbook PDF. Extract nonempty pages with `pypdf`; preserve page numbers/source/title.
2. Split page documents with recursive chunking, default 800 characters and 120 overlap. Add chapter/section when detected, stable chunk identifiers, and content-derived document IDs. Memoize the loaded corpus within the worker.
3. Construct OpenAI dense embeddings, default `text-embedding-3-small`, and local `FastEmbedSparse` with `Qdrant/bm25`.
4. Reuse the configured Finance hybrid collection when nonempty and sparse-compatible; otherwise create dense cosine and sparse vector slots, use IDF sparse configuration, embed/chunk-index the corpus, and retain citation metadata in payloads. Existing nonempty collections are reused rather than automatically reconciled against every source change.
5. On a cache miss, the Finance policy exposes `finance_rag_query`, `finance_expand_evidence`, and `finance_targeted_search` through the common runtime. Finance metadata extraction can constrain search from the task and available corpus metadata.
6. Build baseline or targeted dense subqueries and sparse keywords from model tool arguments; use the single query when no list is supplied. Validate the Finance plan/budget. Expansion retains previous terms and increases retrieval depth.
7. Acquire a Finance retrieval semaphore slot. Run remote dense query embeddings asynchronously; perform local synchronous sparse encoding in `asyncio.to_thread` so it does not occupy the event-loop thread. This does not make CPU execution itself inherently faster.
8. Submit dense and sparse prefetches to Qdrant with payload filters, then combine results using Reciprocal Rank Fusion. Qdrant's synchronous query client is thread-offloaded. Default baseline depth is 10 per dense/sparse search, 20 fused candidates, RRF k=60, and a 15-second retrieval deadline.
9. Convert points into evidence records with content, metadata, score/rank, and provenance. Keep the retrieval ordering when cross-encoder reranking is disabled, which is the default. If enabled, score question/passage pairs using the local sentence-transformers model; cache its loaded instance and serialize native inference with a shared lock. Nonfinite reranker scores fall back to original ordering.
10. Select three baseline or five expansion passages by default, deduplicate across actions, and enforce shared evidence limits. Recover only necessary unsupported facts; sufficient definitions need not spend the unused budget.
11. Model-generated prose is grounded in supplied passages, with handbook page citations. The model submits the final result through the completion tool; shared validation and output checks follow.

## 3. Finance Q&A: calculation branch

1. Identify a simple-interest, compound-interest, or comparison request. The model selects deterministic calculator tools rather than computing amounts in prose.
2. Validate principal, rate representation, duration, compounding mode/frequency, supported currency, contradictions, and unsupported deposits/withdrawals/fees/taxes according to the calculation tools.
3. For missing/unsupported calculation inputs, use the specialized interest clarification tool, which groups unresolved details and enforces its clarification limit. The generic Finance clarification tool redirects numeric interest clarification to that specialized path.
4. Obtain explicit agreement before excluding unsupported features; never silently change the requested calculation.
5. Execute successful calculations and have the result processor bind authoritative amounts to the response. Pure arithmetic bypasses handbook retrieval; explanations or mixed questions still need evidence.
6. For a comparison, calculate both requested methods with the appropriate shared inputs. Preserve full calculation data while formatting user-facing money/assumptions.
7. Return the calculation with the answer; such results are ineligible for the public semantic definition cache.

Source anchors: [Finance graph/policy](../../src/finassist/agents/finance_qa/graph.py), [corpus indexing](../../src/finassist/agents/finance_qa/corpus_index.py), [Finance retrieval](../../src/finassist/agents/finance_qa/retrieval.py), [hybrid query engine](../../src/finassist/rag/retriever.py), [recovery](../../src/finassist/agents/finance_qa/evidence_recovery.py), [calculator tools](../../src/finassist/agents/finance_qa/calculator_tools.py), [reranker](../../src/finassist/agents/finance_qa/reranker.py).

## 4. Tax Education: ingestion and evidence branch

1. Load the packaged expanded IRS tax-education JSONL, not the companion PDF as the active ingestion source. Preserve tax year, filing year, jurisdiction, filing statuses, forms, topic/subtopic, title, and other configured metadata.
2. Split records with 800-character chunks and 120-character overlap; retain source metadata and assign chunk IDs.
3. Build/reuse `tax_education_hybrid` through the shared hybrid-vector-store constructor: OpenAI dense embeddings, local FastEmbed BM25, Qdrant cosine/sparse storage, and metadata payloads. This constructor differs from the Finance-specific one; do not assert identical sparse-IDF configuration.
4. After Tax semantic-cache bypass/miss, the model chooses retrieval, calculation, clarification, recovery, or final submission using the shared runtime.
5. The active prompt requests one search query for simple questions and up to three focused `dense_queries` for multipart questions. This is model-directed decomposition, not a mandatory planner node or deterministic guarantee.
6. The active evidence-recovery tool collects the original task and actual clarification replies, extracts deterministic Tax metadata, and builds filters. Omitted years use the supported 2026 corpus scope; explicit user years must not silently change. Verify the exact filter fields against `extract_tax_metadata` rather than assuming every natural-language constraint is a database field.
7. Pass explicit dense/sparse search inputs to `TaxHybridRetriever`. This bypasses its optional `TaxQueryAnalyzer`. That analyzer and its 1–3-query prompt exist for calls without explicit search inputs, but should not be drawn as an always-executed node in the current agent path.
8. Perform shared dense/sparse embeddings, Qdrant filtered prefetch/RRF, and point-to-document conversion. Recheck explicit tax/filing year on returned documents before evidence is exposed.
9. Default selection is five baseline/targeted passages, ten expansion passages; baseline search requests 20 candidates and expansion 40. The underlying shared hybrid retriever also has its own candidate cap, so requesting 40 does not by itself prove 40 candidates are returned. Optional cross-encoder reranking is disabled by default.
10. Deduplicate and budget evidence. The model checks coverage of every requested aspect, then submits or chooses one bounded expansion/targeted action. Evidence gaps remain insufficient, not user-input clarification or out of scope.
11. Cite only corpus-supplied titles used in support. The knowledge corpus is static packaged material, not a live IRS website verification service.

## 5. Tax Education: calculation and human-input branch

1. General definitions need no personal income/status. A personal bracket lookup needs income and filing status; the active model/tool policy owns clarification, not the bypassed analyzer alone.
2. For federal ordinary-income tax amounts, call `calculate_federal_ordinary_income_tax`. It supports 2026, USD, and taxable ordinary income; salary/gross/take-home values are not silently treated as taxable.
3. Require essential missing values or permission for a simplified calculation when unsupported credits, preferential gains/dividends, payments, AMT, or other features are present. Use `ask_tax_clarification` to pause and resume; never submit a final clarification-shaped object instead.
4. Compute bracket slices and totals deterministically, then validate/expose the trusted calculation. Disclose tax year, income basis, and exclusions; do not call it a completed return or refund/balance-due determination.
5. Questions explicitly prohibiting calculation use only the evidence branch. A multipart reporting question can request forms and tax-treatment explanations without invoking the calculator.
6. Submit `submit_tax_answer`, normalize result consistency, return to outer output checks, and cache only eligible public definitions.

Source anchors: [Tax policy](../../src/finassist/agents/tax_education/graph.py), [Tax corpus/filter/retriever](../../src/finassist/rag/tax_education.py), [hybrid ingestion](../../src/finassist/rag/hybrid.py), [active recovery](../../src/finassist/agents/tax_education/evidence_recovery.py), [optional analyzer](../../src/finassist/agents/tax_education/query_analyzer.py), [Tax calculator tools](../../src/finassist/agents/tax_education/calculator_tools.py).

## 6. Market Analysis: provider discovery and execution

1. Enter the Market clarification child graph. Its run node invokes the Market handler. Disabled quotes produce a disabled-feature answer; a missing Twelve Data key produces an explicit unavailable result.
2. Apply the Market active deadline, default 30 seconds. Borrow a worker-owned MCP connection from `MarketSessionManager`. The owner task starts/owns/cleans up the stdio transport and subprocess; borrower cancellation does not own its lifecycle.
3. Launch the configured official Twelve Data MCP Python command with configured server arguments and key environment. The script is an external pinned runtime installed separately. It calls Twelve Data; FinPilot's main code is the MCP client, not a second implementation of the HTTP provider API.
4. Request MCP tool schemas through cached discovery. Filter to allowed provider tools and add local growth/extrema tools. Extend quote/history descriptors with `finish_response`; model prompts and schemas guide selection.
5. Send the task, current context/time, Market rules, and allowed descriptors to the chat model. Resolve company names/uncertain listings via `search_symbol`; symbol matches must precede choosing an uncertain listing. A provider candidate list and the model's selected ticker are different outputs.
6. Before history/growth/extrema calls, deterministically resolve the timeframe from the original task plus Additional details. Supported explicit dates use YYYY-MM-DD; supported relative windows normalize units/counts/intervals. Ambiguous dates or incompatible windows return a clarification question.
7. On `status=clarify`, the wrapper enters Ask for missing details, interrupts, and checkpoints. Resume appends the answer and re-enters Market without rerouting. Do not depict the resolver's clarification as an empty final answer.
8. For ordinary provider calls, query the process-local TTL/LRU response cache using normalized tool arguments. Reuse a valid hit or share an identical pending task. On a miss, apply provider concurrency/backoff controls and invoke the MCP tool with safe tracing. The request budget defaults to eight fetch operations; retries and in-flight tasks are implementation details, not a proven global hard HTTP-attempt quota across workers.
9. Routine `get_quote` retrieves quote fields and provider timestamps; `get_time_series` retrieves interval-bound history, at most 100 requested records. Symbol/history defaults and explicit window arguments are validated rather than model-invented dates.
10. Parse structured provider payloads; preserve quote metadata and historical observations separately. Missing currency/exchange/time/values remain unavailable. Detect rate-limit/provider errors and avoid caching error payloads.
11. A final routine quote/chart call with `finish_response=true` uses deterministic snapshot formatting and full structured observations without another prose-generation turn. More analysis can continue the model/tool loop within limits.
12. Return `market_observations` calculations for UI validation and charting, along with answer/status. The UI requires NASDAQ/USD for its stock widgets even though the agent supports a broader instrument set.

## 7. Market Analysis: deterministic metric branches

1. For period return or multi-asset performance, choose `calculate_price_growth`; resolve a common explicit/relative timeframe and pass every requested symbol, at most four. Do not substitute quote daily percentage change for period return.
2. Fetch each needed history through the same Market cache/MCP layer; validate observations, shared dates and numeric inputs; compute growth deterministically and disclose coverage and price-only limitations. Same-provider calls are sequential in these calculation paths.
3. For highest/lowest price, select `calculate_price_extrema`, defaulting maximum to daily high and minimum to daily low unless open/close is requested.
4. Validate an inclusive window of at most 100 calendar days and matching valid provider observations. Fetch daily history through the same cache; reject invalid/conflicting values and compute Decimal-based extrema and matching dates.
5. Format extrema prices to two decimal places in answer text while keeping full precision in structured calculation data. Disclose incomplete trading-session coverage/current bars and missing currency if unavailable.
6. Longer windows are currently a limitation. No pagination-based full-year extrema workflow is implemented. Explain limits or clarify rather than claiming complete year coverage.
7. Detailed Market provider trace output currently records error summaries/record counts, not all public symbol-search candidates or raw HTTP calls inside the MCP subprocess. Do not draw nonexistent provider-HTTP spans as already instrumented.

Source anchors: [Market agent](../../src/finassist/agents/market_analysis/agent.py), [stdio connection](../../src/finassist/agents/market_analysis/agent.py), [sessions](../../src/finassist/agents/market_analysis/session.py), [response cache](../../src/finassist/agents/market_analysis/cache.py), [timeframes](../../src/finassist/agents/market_analysis/timeframe.py), [growth](../../src/finassist/agents/market_analysis/growth.py), [extrema](../../src/finassist/agents/market_analysis/extrema.py), [presentation](../../src/finassist/agents/market_analysis/presentation.py), [provider traces](../../src/finassist/agents/market_analysis/tracing.py).

## 8. Portfolio Analysis: scoped synthetic analysis

1. At application startup, seed one in-memory SQLite store with managers, customers, pre-established investment profiles, holdings, costs, and frozen October 9, 2026 reference prices. No live quote provider, RAG index, or brokerage connection participates in Portfolio calculations.
2. Resolve the simulated application identity from backend settings and validated local manager override. Pass context in graph state and `configurable`; customer/manager IDs written in user text are not authentication.
3. The model receives domain/system guidance, the task and requested scope, and runtime application identity. Tools instantiate a scoped `PortfolioTools` from runtime context. If neither runtime nor configured identity exists, return access denied.
4. For a manager-wide dashboard/listing, the prompt directs `screen_manager_portfolios(screen=all)` with omitted manager ID; the tool derives the manager from identity. For a customer's own account, `get_customer_portfolio` can omit customer ID and derive it from CUSTOMER identity. These choices remain model-directed; no deterministic dashboard dispatcher exists.
5. A manager requesting a specific customer supplies/resolves that customer in authorized scope. `resolve_portfolio_customer` searches allowed names and handles multiple matches through clarification. A manager-wide listing does not require a single customer ID.
6. `get_customer_portfolio` validates access and reads holdings, then deterministically computes value, holding/sector/asset allocation, concentration, HHI/effective holding count, unrealized performance when cost basis permits, and gainers/losers.
7. `get_customer_investment_profile` retrieves stored profile facts; the model requests it alongside metrics only for profile alignment. The model cannot assign a new official risk profile or infer a suitability certification.
8. `screen_manager_portfolios` screens authorized customers for all/concentrated/single-stock conditions; its tool default remains concentrated, so all-customer listing requires explicit `screen=all`. `compare_manager_portfolios` compares selected/all assigned customers with supported concentration/diversification/unrealized-loss ranking. Incomplete accounts are excluded with reasons where needed.
9. Reject cross-scope IDs at the tool boundary; do not retry to evade ACCESS_DENIED. This is implemented simulated scope enforcement, not real authenticated production authorization.
10. A successful single routine screen/comparison call with `finish_response=true` can trigger deterministic `_CompletedScreen` formatting before another model turn. The shortcut requires exactly one qualifying call in history; extra tool calls prevent that shortcut. The current formatter still exposes technical screen/ranking/source labels; a more reader-friendly formatter was proposed but not implemented.
11. Otherwise a structured formatter assesses sufficient/aligned coverage of the request using actual tool messages. If incomplete, the loop may call additional tools or ask an essential clarification, within six model decision turns and a 60-second default deadline. These limits also apply across resumed model messages.
12. Return successful calculation payloads and answer, preserving frozen-price/synthetic provenance. UI adapters display validated snapshots, customer directory, Summary, Positions, and Allocation. No semantic answer cache, provider price refresh, trade execution, or persisted portfolio editing exists.

Source anchors: [seed store](../../src/finassist/data/portfolio_store.py), [agent](../../src/finassist/agents/portfolio_analysis/agent.py), [tools/calculations](../../src/finassist/agents/portfolio_analysis/tools.py), [authorization](../../src/finassist/agents/portfolio_analysis/authorization.py), [UI snapshots](../../src/finassist/web_app/services/portfolio.py).

## 9. Goal Planning: deterministic projection with model-selected operations

1. Receive the savings task; use no corpus retriever, vector database, market/news provider, or semantic answer cache.
2. Start the Goals child graph with a ReAct tool loop, active deadline 60 seconds and recursion limit 30 by default. Expose required-contribution, future-value, time-to-goal, feasibility, and clarification tools.
3. The model determines which operation the user needs. Required contribution needs target/horizon; future value needs contribution/horizon; time-to-target needs target/contribution; feasibility needs target/contribution/horizon.
4. Normalize units faithfully; omitted starting savings and nominal annual return default to zero. Ask grouped missing/ambiguous essentials with `ask_goal_clarification`, producing a real graph interrupt. Resume retains target/purpose and supplied replies.
5. Deterministic calculators reject invalid inputs and use end-of-month contributions with monthly compounding. Horizon must be a positive whole number of months, at most 100 years; rates are percentage inputs. Exclusions include taxes, fees, inflation, withdrawals, and variable/downside return modeling.
6. Calculate projected future value, required monthly savings rounded up to cents, time in whole months, or feasibility/shortfall/surplus as selected. An unreachable goal is a valid result, not a transport error.
7. A structured assessment checks operation, inputs, assumptions, and request coverage, not independent model arithmetic. Incomplete requests continue the same loop when budget permits.
8. Enforce at most three model decision turns, including clarification. The final coverage check can explain the third turn's result but cannot initiate a fourth decision turn. Expose errors/insufficiency instead of inventing missing math.
9. Validate successful tool outputs and assemble calculation payloads and assumptions. Final explanation uses those values; it does not invent citations, tax rules, real returns, or guaranteed outcomes.
10. Return to shared output checks and UI. Separately, the Goals UI's local chart uses `project_goal`; drawing that chart alone does not execute this agent.

Source anchors: [Goals agent](../../src/finassist/agents/goal_planning/agent.py), [calculators](../../src/finassist/agents/goal_planning/tools.py), [local UI projection](../../src/finassist/web_app/services/showcase.py).

## 10. News: evidence acquisition and supported synthesis

1. Receive either a chat question or the News UI's generated topic/period request. News is not the historical price-return calculator. Topic content may discuss a year while the requested article window is a week.
2. Enter the News child graph with a 60-second active deadline and recursion limit 30. Acquisition uses a chat model with `search_financial_news` and `ask_news_clarification`; parallel tool calls are disabled and only one selected call is retained per acquisition decision.
3. `understand` chooses an initial search or essential scope clarification. Clarification interrupts and resumes through the parent checkpointer. The initial model may return no tool call, ending acquisition; final validation must not treat that as evidence.
4. `count_search` increments the bounded search count. The direct Tavily tool builds HTTPS `POST /search` with bearer authentication, `topic=news`, basic depth, max three results, and provider answer generation disabled.
5. The tool uses a rolling day/week/month/year parameter, default week, unless both valid explicit ISO date bounds are supplied. Explicit dates override the rolling filter; the code advances the inclusive requested end day to the provider's exclusive upper bound. The UI's textual Today choice still depends on model tool arguments, not a typed enforced publication-window field.
6. Validate nonempty query, date inputs, configured key, HTTP outcome, URL/metadata/content records, and construct evidence IDs from tool execution history. Missing publication dates stay missing. Returned provider evidence is data, not instructions.
7. Tool execution returns content plus an evidence artifact. A successful retrieval enters `assess`, whose structured model evaluates actual coverage, missing aspects, and an optional next focused search.
8. If sufficient, stop acquisition immediately. If insufficient and below three searches, use a proposed focused query when available, preserving prior explicit timeframe arguments. Otherwise stop with available evidence. There is no promise of three searches on every request.
9. Gather evidence and safe errors from tool history. Deduplicate/validate evidence IDs and detect conflicting ID-to-URL mappings.
10. A separate structured formatter produces claims with evidence IDs and reported-fact versus interpretation labels. News assessment and formatting use tool-based structured output to avoid the problematic parsed-response serialization path.
11. Python checks claim references against retrieved evidence, omits unverifiable claim references, constructs source URLs itself, and labels interpretation. This enforces reference provenance, not a deterministic proof that every sentence is entailed by the article.
12. Missing/partial coverage or omitted claims yields an explicit insufficient/partial summary. Retrieval failure without usable evidence yields error. A completed formatter clarification result is not exposed as a fake resumable thread.
13. Build `news_articles` structured payloads only for supported reported-fact claims, with original URLs, titles, available publisher/date, and summary text. UI extracts validated article cards and lets Ask about this prefill a follow-up.
14. No Qdrant RAG collection, portfolio SQLite query, MCP transport, or application-level persistent news response cache is used in this path. Provider date filtering is requested, but the current code does not independently enforce a strict UI-selected publication-date window on every displayed article; this explains the observed Today/older-article gap.

Source anchors: [News acquisition/synthesis](../../src/finassist/agents/news_synthesizer/agent.py), [Tavily tool](../../src/finassist/agents/news_synthesizer/tools.py), [evidence/claim schemas](../../src/finassist/agents/news_synthesizer/schemas.py), [UI news adapter](../../src/finassist/web_app/services/news.py).

## 11. Diagram construction handoff order

1. Draw the system boundary and UI-to-API sequence first, with separate new-question and clarification-resume edges.
2. Add the shared input guards, router, six exclusive specialist branches, graph output guard, and API response validation.
3. Draw each agent as its own expanded sequence from this document, keeping provider and database ownership explicit.
4. Attach Qdrant knowledge and answer-cache collections to Finance/Tax only; attach synthetic SQLite to Portfolio; attach checkpoint SQLite to parent orchestration, not to one specialist alone.
5. Attach Market cache/session/MCP/Twelve Data in their actual order; attach News directly to Tavily, not through MCP.
6. Put LangSmith and console logs as cross-cutting observation paths. Put evaluation tooling outside the runtime request flow.
7. Annotate defaults, optional reranking/cache switches, LLM choices, and unimplemented guarantees. Do not convert a prompt instruction into a deterministic decision node.
