# AI Finance Assistant

A Python 3.12 FastAPI service using LangGraph to classify and route finance questions. Finance education questions can use hybrid retrieval over the bundled handbook; most other intent handlers are still demonstrations with fixed responses.

## Features

- FastAPI application with health and query endpoints
- LangGraph workflow for intent classification and routing
- OpenAI structured-output intent classification
- PDF loading, chunking, metadata extraction, and dense+sparse hybrid retrieval for finance Q&A
- Persistent local Qdrant storage for the handbook index
- Typed graph state, settings, and retriever interfaces

## Prerequisites

- Python 3.12
- A local virtual environment
- An OpenAI API key for intent classification, dense embeddings, and finance Q&A answer generation

## Getting started

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Create a `.env` file and set `OPENAI_API_KEY`. The key is required for query graph initialization, dense embeddings, and LLM-generated answers. The first query may load and index the handbook if the local Qdrant collection is missing or empty.

## Running the app

```bash
source .venv/bin/activate
uvicorn ai_finance_assistant.main:app --reload
```

Then open:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## Query endpoint

`POST /query` accepts a JSON payload like:

```json
{
  "query": "What is an ETF?"
}
```

Response shape:

```json
{
  "intent": "finance_qa",
  "answer": "..."
}
```

The graph is initialized lazily on the first query and cached for the lifetime of the process after successful construction. It creates the intent classifier, answer LLM, and finance retriever. If `OPENAI_API_KEY` is missing, initialization raises HTTP 503. If another required component fails to initialize, graph creation fails and is retried on a later request; a partially initialized graph is not cached. The health endpoint does not require an API key.

## Configuration

The project loads settings from `.env` and environment variables via Pydantic settings.

Key settings include:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `INTENT_ROUTER_SYSTEM_PROMPT`
- `QDRANT_MODE` (`local`, `memory`, or `server`)
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`

The generic Qdrant client helper supports `QDRANT_MODE=local` or `server`; `memory` is accepted as an alias for `local`. The finance handbook ingestion and retriever currently use local Qdrant directly, at `.qdrant`, with collection `financial_foundations_hybrid`. Therefore, `QDRANT_MODE=server`, `QDRANT_URL`, and `QDRANT_COLLECTION` do not redirect that finance Q&A path yet. Local embedded Qdrant storage is intended for one process at a time; use a Qdrant server for concurrent workers or clients.

## Supported intents

The graph currently routes to the following intents:

- `finance_qa`
- `portfolio`
- `market`
- `goals`
- `news`
- `tax`
- `clarify`
- `other`

The system is designed to detect vague or under-specified requests and route them to `clarify` instead of guessing. For example:

- "What is the stock price?" -> clarify if the ticker is missing
- "top news" -> clarify unless the news topic is specified
- "tax details" -> clarify unless the tax question includes the necessary facts

## Current functional and technical status

### Implemented

- FastAPI exposes `/health` and `/query`. The query endpoint validates and trims input and returns an intent with an answer.
- An OpenAI chat model classifies intent using structured output. LangGraph routes `finance_qa`, `portfolio`, `market`, `goals`, `news`, `tax`, `clarify`, and unsupported requests.
- Finance Q&A can load `src/ai_finance_assistant/rag/finance-qa/Financial_Education_Handbook_integrated.pdf`, split it into overlapping chunks, and attach document, chapter, section, page, and chunk metadata.
- Hybrid Qdrant indexing uses OpenAI `text-embedding-3-small` vectors and FastEmbed `Qdrant/bm25` sparse vectors. A populated collection is reused; a missing or empty collection is built from the handbook. Run the ingestion and sample-search script directly with `python -m ai_finance_assistant.agents.financial_qa_ingest`.
- With retrieved context, the answer model is instructed to ground responses in that context and cite source locations when available. If retrieval or answer generation fails or returns no matches, finance Q&A falls back to a fixed educational response.
- Graph and retriever factories cache successful instances per process. Failed graph initialization is not cached.

### Placeholder behavior and limitations

- Portfolio analysis, market data, savings-goal calculations, news, and tax responses are fixed examples; they do not call external domain data sources or perform real analysis.
- Clarification uses simple phrase checks, not a general completeness or fact-validation system.
- Finance Q&A currently uses one bundled PDF and local embedded Qdrant storage. The generic Qdrant helper's server mode is not wired into this path.
- The finance ingestion code currently selects `text-embedding-3-small` directly; `OPENAI_EMBEDDING_MODEL` is not yet applied to that pipeline.
- Local `.qdrant` storage cannot be opened concurrently by multiple Qdrant client instances or server workers. Use a Qdrant server when concurrent access is required.
- A valid `OPENAI_API_KEY` and OpenAI connectivity are required to initialize the query graph. Dense embedding and answer-generation calls use OpenAI; sparse BM25 embeddings are computed locally by FastEmbed.

Next steps include real market and news integrations, actual portfolio and goal calculations, authoritative tax sources, configurable Qdrant server support for finance Q&A, and deployment-oriented observability and error handling.

## Project layout

```text
src/ai_finance_assistant/
  agents/         Specialized agent implementations
  api/            FastAPI routes and request models
  core/           configuration, logging, and shared infrastructure
  graph/          LangGraph state, classification, and builder logic
  integrations/   third-party API client boundaries
  main.py         app bootstrap
  providers/      model/provider abstractions
  rag/            retrieval contracts and Qdrant helpers

tests/
  test_graph.py
  test_health.py
  test_qdrant.py
  test_retriever.py
  test_query_api.py
```

## Quality checks

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
pytest --cov
```

Run the project test suite with `pytest -q`. Ruff, mypy, and coverage configuration is defined in `pyproject.toml`.