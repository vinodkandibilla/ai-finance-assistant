# API and operations

FastAPI exposes `/query`, `/query/resume`, `/health`, and `/portfolio/managers`. See the generated contract at `docs/api/finassist-openapi.json`.

Configure OpenAI, Qdrant, market/news providers, tracing, and checkpoint paths through `.env`. Health is a lightweight process check; it does not validate provider quota, retrieval collections, or model availability.
