# API and local operations

For first-time AWS installation, rsync updates, systemd services and MCP setup without Docker, follow the [EC2 deployment guide](ec2-deployment.md).

Run the application from the repository root with Python 3.12. The [main README](../README.md#quick-start) covers installation; this guide covers provider setup, the HTTP contract, storage and troubleshooting. Configuration defaults are defined in [application settings](../src/finassist/core/config.py), [Finance settings](../src/finassist/agents/finance_qa/config.py) and [Tax settings](../src/finassist/agents/tax_education/config.py).

## Provider setup and startup

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Set `TWELVE_DATA_API_KEY` for Market and `TAVILY_API_KEY` for News. Keep credentials out of source control and screenshots.

Market uses a separately installed Twelve Data MCP subprocess. Install its pinned environment once:

```bash
.venv/bin/python scripts/setup_twelve_data_mcp.py
```

This downloads the server and installs its dependencies under `.cache/twelvedata-mcp`. It is not bundled into the application wheel. The default executable and arguments use POSIX paths; adapt `TWELVE_DATA_MCP_COMMAND` and `TWELVE_DATA_MCP_ARGS` if the deployment layout differs. See the [setup script](../scripts/setup_twelve_data_mcp.py).

Run the backend and UI in separate terminals:

```bash
.venv/bin/python -m uvicorn finassist.main:app --host 127.0.0.1 --port 8000
.venv/bin/python -m streamlit run src/finassist/web_app/app.py
```

Use one backend worker with embedded Qdrant. Development can add `--reload`; use a source-stable process for demonstrations. Initial indexing and local model downloads require network access. `/health` checks the process, not provider quota, credentials or retrieval readiness.

## Configuration

Settings load environment variables and the repository `.env`; explicit process environment variables take precedence. Restart the backend after changing cached settings or prompts.

| Setting | Source default and purpose |
| --- | --- |
| `OPENAI_MODEL` | `gpt-4o-mini` for routing and most specialist reasoning |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` for dense embeddings |
| `INTENT_ROUTER_SYSTEM_PROMPT` | Optional override of the built-in router prompt; remove an obsolete override before comparing source changes |
| `ROUTER_TIMEOUT_SECONDS` | 10 seconds for classification |
| `MODERATION_ENABLED`, `INJECTION_CLASSIFIER_ENABLED` | Both true; control input API checks |
| `QDRANT_MODE` | `local`; `server` uses `QDRANT_URL` and optional `QDRANT_API_KEY` |
| `FINANCE_COLLECTION` | `finance_education_hybrid` |
| `FINANCE_CHECKPOINT_PATH` | `.cache/finance_agent_checkpoints.sqlite`, shared checkpoint database across domains |
| `PORTFOLIO_SIMULATED_USER_ID`, `PORTFOLIO_SIMULATED_USER_ROLE` | `PM001`, `PORTFOLIO_MANAGER`; simulated application identity, not real authentication |
| `FINANCE_AGENT_MAX_SEARCHES`, `TAX_AGENT_MAX_SEARCHES` | Three total retrieval actions per domain request |
| `FINANCE_RERANKER_ENABLED`, `TAX_RERANKER_ENABLED` | False by default |
| `MARKET_AGENT_TIMEOUT_SECONDS` | 30 seconds overall; per-MCP-call timeout remains deferred |
| `MARKET_PROVIDER_MAX_RETRIES` | Two retries after the initial attempt, transient failures only |
| `MARKET_CACHE_MAX_ENTRIES` | 100 entries in the process-local TTL/LRU cache |
| `TAVILY_TIMEOUT_SECONDS` | 15 seconds per HTTP search |

Finance and Tax settings also read `.env`, using their respective prefixes. Their source files define the complete tunable surface, including passage limits and cache settings; this table lists the common operations controls.

## HTTP contract

The [OpenAPI snapshot](api/finassist-openapi.json) defines wire schemas. The running service exposes `/docs`; the [portfolio dashboard contract](api/portfolio-dashboard.md) adds snapshot validation details.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Lightweight process health |
| `GET /portfolio/managers` | Allowed simulated manager directory |
| `POST /query` | Start a new question with a new thread |
| `POST /query/resume` | Resume a real interrupted task using its returned thread ID |

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is an ETF?"}'
```

For a response with `status="clarification"` and a real `thread_id`, send the reply to `/query/resume`:

```bash
curl -X POST http://127.0.0.1:8000/query/resume \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"<returned-thread-id>","reply":"Five years"}'
```

Do not reuse that example placeholder as a real thread. Router-level `clarify_node` is a completed response without a resumable thread. New questions do not automatically receive prior UI chat messages. Request scope fields cannot establish identity; the backend supplies simulated context and checks resume ownership.

| HTTP status | Meaning |
| --- | --- |
| 200 | A response was returned; inspect its application status for answered, insufficient, blocked, clarification or domain error outcomes |
| 422 | Request validation failed; new questions allow 1–2,000 characters and resume replies 1–4,000 |
| 403 | Resume thread is outside the current simulated identity scope |
| 502 | Invalid classifier output |
| 503 | Classification provider unavailable, or a required service dependency is unconfigured |
| 504 | Classification deadline exceeded |

Classifier error responses contain a safe `detail`, not a fabricated intent or handler. Other caught execution errors can still produce HTTP 200 with `status="error"`. The UI handles HTTP failures without automatically retrying. A client timeout does not establish whether server execution completed.

## Storage and index lifecycle

| Data | Location and lifetime |
| --- | --- |
| Finance and Tax retrieval indexes | Qdrant collections `finance_education_hybrid` and `tax_education_hybrid` |
| Definition answer caches | Separate Qdrant collections `finance_answer_cache` and `tax_answer_cache`, one-hour default TTL |
| Embedded Qdrant | Checkout `.qdrant/` with the current package layout; installed-wheel paths can differ |
| Clarification state | File-backed SQLite at `FINANCE_CHECKPOINT_PATH` |
| Synthetic portfolios | Separate seeded SQLite `:memory:` database, recreated at startup |
| Market responses | Process-local TTL/LRU cache, lost on worker restart |
| UI conversation and retained results | Streamlit session state; not durable account history |

The first graph dependency use can initialize Finance and Tax indexes. Populated collections reuse document vectors. Missing or empty collections can trigger embedding work. There is no source fingerprint check or automatic refresh when corpus files change, and no general rebuild CLI.

After changing a Finance corpus, chunking or embedding model, stop the API and select a new `FINANCE_COLLECTION` to rebuild dense and sparse vectors together. The Tax collection name is fixed in code; rebuild it deliberately with the application stopped and preserve needed data first. Do not delete collections while a local client holds the storage lock. Legacy dense-only collections do not satisfy the hybrid schema.

Qdrant persistence reuses document embeddings; it does not eliminate per-search query embeddings. The MCP subprocess environment, checkpoint database, semantic caches and Market response cache have separate lifecycles.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Health works, queries fail | Provider credentials/quota, first-use indexing, MCP installation and server logs |
| Routing differs from source or saved evals | Exact submitted query, UI wrapper, prompt override, model and process restart |
| Market tool startup fails | Run the setup script; verify the configured MCP executable and server paths |
| Retrieval schema mismatch | Use the expected hybrid collection and compatible embedding configuration |
| Changed corpus produces old answers | Rebuild deliberately and account for semantic answer-cache namespaces/expiry |
| Portfolio content or chat disappears | Inspect the first page-rendering exception, including NumPy/PyArrow compatibility; changing routing will not fix an import error |

Use [evaluation and observability](evaluation.md) for correctness and latency checks. Deployment across multiple workers or servers needs an explicit storage, identity and provider-limit design; the local setup does not establish production readiness.
