# FinPilot system architecture: ordered execution specification

This document describes the current source tree, including uncommitted changes, reviewed on October 27, 2026. It is a specification for drawing architecture diagrams later, not a deployment claim. Read it first, then [router execution](02-router-sequence.md), then [specialist internals](03-agent-sequences.md). No diagram is included.

“User” means the person using Streamlit. “Model” means an external OpenAI model invoked through LangChain/OpenAI clients. “Tool” means a callable available to an agent; only Market uses an MCP transport. “Human in the loop” means checkpointed clarification, not human approval of every answer or trade. “Default” means source-code defaults, which environment configuration can override.

## 1. Runtime preparation and application construction

1. Install the Python 3.12 application and its compatible dependencies on the host. Prepare the NLP model, local embedding/reranking assets where needed, and the separately pinned Twelve Data MCP runtime using `scripts/setup_twelve_data_mcp.py`. The MCP subprocess environment is not bundled into the application wheel. Missing dependencies can surface on first use rather than at process startup.
2. Provide backend settings and provider keys. The application settings read `.env`; LangSmith requires its settings in the process environment, for example via Uvicorn's `--env-file`. Do not equate tracing tags being enabled with tracing transport being enabled.
3. Start FastAPI. Its lifespan opens an `AsyncSqliteSaver` checkpoint database, calls saver setup, creates a synthetic `PortfolioStore`, constructs simulated portfolio identity from settings, and creates the Market session manager. It closes these owned resources on shutdown.
4. Keep the two SQLite purposes separate. Checkpoints are file-backed, defaulting to `.cache/finance_agent_checkpoints.sqlite`, and support the parent graph across domains. The portfolio store is a separate seeded SQLite `:memory:` connection, not a persistent customer database.
5. Resolve the cached `get_query_graph` dependency when requested. It constructs the router, shared OpenAI chat client, Qdrant client, retrieval components, Finance/Tax semantic caches when enabled, six specialists, and the compiled parent LangGraph with its checkpointer. This dependency is cached in the application process; it is not a new graph build on every question.
6. During retrieval construction, reuse nonempty hybrid collections or initialize missing/empty collections from packaged corpora. This can perform disk reads, local sparse encoding, and remote embedding calls before agent execution. It is not a separately deployed ingestion service.
7. Start Streamlit as a separate process. Its HTTP client points to FastAPI via `FINPILOT_API_BASE_URL`. Streamlit does not read portfolio SQLite or Qdrant directly.
8. Treat `/health` as process health only. It does not prove model quota, corpus coverage, MCP installation, or provider availability. There is no comprehensive readiness gate yet.

Source anchors: [FastAPI lifespan](../../src/finassist/main.py), [graph dependency and API](../../src/finassist/api/query.py), [settings](../../src/finassist/core/config.py), [Qdrant modes](../../src/finassist/rag/qdrant.py), [MCP installation](../../scripts/setup_twelve_data_mcp.py).

## 2. User entry and UI paths

1. Streamlit creates sidebar navigation for Overview, Finance Q&A, Portfolio, Market, Goals, News, and Tax Education. The selected page occupies the content pane; the shared Ask FinPilot conversation occupies the assistant pane. Page rendering precedes chat rendering, so an uncaught content-rendering error can prevent chat from appearing.
2. On Overview, selecting an Explore card opens a workspace and prefills a suggested question. Finance/Tax learning questions similarly prefill the composer for editing and explicit sending.
3. In chat, retain display messages, pending clarification thread, failed request, and drafts in Streamlit session state. Navigation shares that conversation during the session. This is not durable chat history, and ordinary new API questions do not include the complete visible history.
4. Portfolio first requests `/portfolio/managers`. The API returns a context-dependent manager directory and whether switching is permitted. A local-only manager header is validated by the server; a request-body role or ID is requested scope, not trusted identity.
5. Portfolio dashboard loading uses a generated `/query` request. With a manager override the request is manager-wide; otherwise it asks for the dashboard available to the current account. Selection also scopes subsequent chat calls through the manager header. This is still LLM-routed retrieval, not a deterministic dashboard endpoint.
6. Market controls issue `/query` requests for quotes/history under the explicit NASDAQ/USD UI scope. Structured observations, not answer prose, feed quote cards and OHLC validation. For a single unambiguous NASDAQ/USD ticker returned by the server, the UI updates the input and reuses that response. Comparison and watchlist actions use their own service calls; watchlists are session-only.
7. Goals controls compute a local hypothetical projection immediately. This path has no FastAPI, LangGraph, or OpenAI call. The Ask AI action submits the scenario inputs through the normal API flow.
8. News controls choose Category, Published within, and an optional custom topic. The custom topic overrides the category. The service wraps them into a news request string and calls `/query`. The period is currently textual input to the agent, not a strongly typed API filter. Previous results can remain visible after a failed search, with a notice.
9. UI HTTP adapters deserialize typed responses, validate structured data, and present answers, statuses, sources, and calculation details. Portfolio validates synthetic provenance and consistent arithmetic before drawing tables/charts; Market validates exchange, currency, ticker, dates, and OHLC ranges before drawing candles.
10. Retry resends the stored failed request. Editing restores text for correction. Pending clarification changes the composer into a resume reply. New Chat clears UI conversation/draft state; it is not a database-retention mechanism.

Source anchors: [app shell](../../src/finassist/web_app/app.py), [chat state and submission](../../src/finassist/web_app/components/chat.py), [HTTP client](../../src/finassist/web_app/services/api_client.py), [showcases](../../src/finassist/web_app/components/showcase.py), [portfolio adapter](../../src/finassist/web_app/services/portfolio.py), [news adapter](../../src/finassist/web_app/services/news.py).

## 3. New request: end-to-end sequence

1. The UI sends JSON to `POST /query`, including the question and optional requested scope. It may send `X-FinPilot-Source` and the local manager-selection header. FastAPI/Pydantic validates supported fields and limits; source labels are normalized for tracing.
2. FastAPI resolves application-owned graph/checkpointer/store/session dependencies and server-supplied simulated identity. Invalid or unauthorized manager overrides fail at this boundary.
3. A `FinPilot request` tracing parent captures a PII-redacted question preview, not graph/client objects or credentials. Tracing must be configured independently; it is not a prerequisite for answering.
4. The handler generates a thread UUID, redacts the question, and invokes the parent graph with canonical `query`, thread ID, requested scope, and portfolio context. Identity is also passed under `configurable` for tool enforcement. Redundant `user_query` and `search_query` state channels have been removed.
5. `input_guardrail_node` redacts again, then runs deterministic regex checks, optional OpenAI moderation, and optional semantic injection classification in that order. Regex rejection prevents those later model calls. A blocking result sets status and routes directly to the graph output guard.
6. Allowed input reaches intent classification. The router assigns one domain and its handler; it does not fan the question out across all specialists. A standalone greeting can finish without a specialist. See the router specification for branch details.
7. Finance/Tax handler wrappers check their semantic answer caches before executing their agents. Cache eligibility is narrow; a hit bypasses that specialist's reasoning/retrieval, not the router or outer output guard. Other agents do not use those semantic caches.
8. The chosen specialist performs its internal sequence: retrieval and recovery, deterministic calculation, scoped portfolio queries, MCP/provider calls, or news search and synthesis. Optional clarification suspends this sequence rather than pretending to be a completed answer.
9. A completed specialist returns an answer, status, available citations, and domain calculation payloads. Not every result has every field. Typed state is not a complete security boundary or a promise that every structured field has been redacted.
10. `output_guardrail_node` redacts final answer text and handles blank successful answers. It does not recursively sanitize every citation/calculation field. It is not a general output-moderation service or semantic fact-checker.
11. The API extracts a real clarification interrupt if present; otherwise it constructs a completed response. It redacts the displayed answer again, validates the public response, normalizes invalid orphan clarification fields, and gives insufficient empty outcomes a readable message. A valid paused clarification is preserved even if the parent answer is initially empty.
12. The API emits its output-check span and available usage summary under the request trace. The cost summary only aggregates usage available in returned model-message metadata; it is not a complete accounting ledger for all nested calls.
13. Streamlit renders the response. A domain error result can arrive over HTTP 200; transport status and domain success are different. Connection/HTTP/contract failures use UI retry/edit controls; answered/insufficient results are ordinary displayable outcomes.

Source anchors: [request/response construction](../../src/finassist/api/query.py), [graph and guard ordering](../../src/finassist/workflow/builder.py), [canonical state](../../src/finassist/workflow/state.py), [PII policy](../../src/finassist/guardrails/pii.py).

## 4. Human-in-the-loop clarification and resume

1. A specialist detects an essential missing user input. It must distinguish missing user information from missing evidence. Lack of corpus support is not a reason to ask the user to supply the answer's facts.
2. Finance, Tax, Goals, Portfolio, and News expose clarification tools that call LangGraph `interrupt` with `kind=clarification` and a question. Market instead returns `status=clarify`; its child wrapper persists the domain result, enters an explicit clarify node, and calls the same interrupt mechanism.
3. Parent/nested graph execution pauses and the SQLite checkpointer retains execution state. A `GraphInterrupt` in a child trace can represent this expected control flow, not an application crash.
4. The API returns a clarification question, `status=clarification`, and the thread ID only when a resumable interrupt exists. Merely returning `clarification_question` from a final model result is insufficient.
5. The UI retains the thread and asks the user to reply. It restricts actions that could conflict with the pending task. It does not silently start another independent domain request in that thread.
6. The UI posts `thread_id` and `reply` to `/query/resume`. The API retrieves the checkpoint and checks its portfolio context against the current server identity before resuming. This is context matching, not production-grade authenticated ownership.
7. The API redacts the reply, invokes `Command(resume=...)` with the same thread/context, and sets the Finance/Tax cache-bypass flag. The paused specialist continues without re-running the router and retains supplied details through its checkpointed messages. Market appends replies as Additional details to its task.
8. Missing details may cause another bounded clarification. Completion returns a normal answer without a pending thread. Education agents normalize completed insufficient results so stray clarification text is not exposed as a resumable question.
9. Initial input guards are not fully rerun over the resume reply. Resume-path guardrail parity is a known limitation. Do not draw the resume edge as necessarily traversing every initial guard again.
10. No trade approval or human financial-advice approval workflow exists. No real order execution exists.

Source anchors: [interrupt helper and Market wrapper](../../src/finassist/agents/common/clarification.py), [resume handler](../../src/finassist/api/query.py), [education runtime](../../src/finassist/agents/common/runtime.py).

## 5. Data stores and cache lookup sequences

1. Qdrant local mode opens persistent storage at the package-derived `.qdrant` path; server mode connects to a configured Qdrant URL/key. These are alternative modes, not two sequential databases. In an installed wheel the derived local path may be under the installation tree: do not assume it is always the working directory.
2. Finance knowledge uses `finance_education_hybrid` by default; Tax uses `tax_education_hybrid`. They contain dense/sparse vectors, passages, and citation/filter metadata. They are knowledge indexes, not chat-history stores.
3. Finance answers use `finance_answer_cache`; Tax answers use `tax_answer_cache`, each a separate Qdrant collection. Eligibility first checks narrow context-independent concepts. Noneligible requests bypass lookup.
4. An eligible lookup tries an exact normalized-key/namespace entry, checking expiry. On a miss it embeds the question, then searches by vector with namespace, concept, and unexpired-entry filters. The similarity threshold is 0.92 for Finance and 0.90 for Tax by default. Similarity alone is not sufficient for reuse.
5. A miss computes the specialist result. Only successful, nonempty, cited, approved-source outcomes without errors, pending clarification, calculations, or interrupts are cacheable. A successful write also cleans expired entries. Default answer TTL is one hour. Cache exceptions fall through to computation.
6. Namespace hashes include prompt/model/embedding/configuration/corpus inputs. They separate incompatible releases; they are not user authorization. Only public definitions are cached, not personalized financial answers.
7. Market's TTL/LRU cache is in process memory. Lookup keys are tool name plus normalized arguments. Quotes live 30 minutes; symbol search and time series live 24 hours. Identical in-flight requests share a task; error responses are not deliberately cached. Restarting the worker loses these responses.
8. Market tool discovery has a separate session/tool-catalog TTL, default 30 minutes. Reusing discovered schemas or an open connection does not mean quote data is cached there.
9. `lru_cache` reuses settings, graph dependency, Qdrant client, Finance corpus loading, and optional cross-encoder instances in process memory. Library model downloads can occupy disk caches, but the application does not implement a general disk-backed HTTP/news response cache or a dedicated dense-query embedding cache.
10. Distinguish persistent Qdrant answer entries from persistent corpus indexes, model-download files, and persistent SQLite checkpoints. All use storage, but they solve different problems. UI session-held last results are another, separate reuse mechanism.

Source anchors: [semantic cache](../../src/finassist/agents/common/answer_cache.py), [Market response cache](../../src/finassist/agents/market_analysis/cache.py), [session/tool discovery](../../src/finassist/agents/market_analysis/session.py), [corpus loader](../../src/finassist/agents/finance_qa/corpus_index.py).

## 6. Cross-cutting boundaries and operational side paths

1. OpenAI chat provides routing and agent reasoning; OpenAI embeddings provide dense vectors; OpenAI moderation is a separate optional HTTP service. Local FastEmbed BM25 supplies sparse vectors. Optional sentence-transformers cross-encoding runs locally. Do not combine these into a single LLM box with identical data flows.
2. Only Market invokes the official Twelve Data MCP server: FastAPI worker to local stdio subprocess, then provider network calls inside that runtime. News calls Tavily directly over HTTPS. Finance/Tax query Qdrant directly; Portfolio queries local SQLite; Goals calculates locally.
3. LangSmith receives configured parent/child traces, model/tool metadata, and redacted parent previews. Provider tracing intentionally summarizes Market responses rather than exposing all raw candidate listings. Child messages/structured payloads do not inherit a blanket recursive PII guarantee.
4. Console logging is separate from trace export. CloudWatch integration, load balancers, autoscaling, managed identity, external checkpoint databases, and Docker deployment are not implemented by this application merely because they were discussed. Local and single-EC2 operation are deployment arrangements around these processes.
5. Bounded execution, asynchronous I/O, thread-offloaded blocking work, caches, and provider concurrency controls improve workload efficiency. They are not measured capacity guarantees or distributed scaling. Most reuse/limits are worker-local; SQLite/local Qdrant require care before adding workers.
6. Evaluation is a development side path, not an evaluator service called on every API response. Router A/B datasets/reporting and Finance grounding/calculation/recovery tests exercise behavior offline or through configured evaluation runs; LangSmith publishing/tracing supports inspection. Runtime evidence assessment is a different mechanism.
7. Current PII allowlist protects selected personal identifiers; driver-license detection was deliberately removed because of false positives. Regex competitor blocking still includes Robinhood and can reject valid stock research before routing. These policies are implementation facts, not recommendations for a production deployment.
8. The system depends on compatible NumPy/spaCy/PyArrow/SciPy packages. A UI rendering failure can precede chat creation; startup success does not validate every native dependency. Record dependency/readiness work as remaining operational work rather than an already deployed safeguard.

Continue with [router execution](02-router-sequence.md), [agent internals](03-agent-sequences.md), and [review findings](04-review.md).
