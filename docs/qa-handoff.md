# FinPilot MVP QA handoff — October 14, 2026

## Automated validation

- Full backend/frontend suite: **616 passed**, 102 existing LangChain/LangGraph
  deprecation warnings. Coverage: **89.84%** (80% required).
- Frontend + UI tests + smoke script: Ruff lint/format pass.
- Frontend strict mypy: pass, 23 source files.
- `pip check`: no broken requirements. `git diff --check`: pass.
- Repository-wide checks still report existing issues outside this UI work:
  13 Ruff findings, 188 mypy errors in 31 backend files, 9 formatting failures.
  These were not suppressed or repaired by changing backend behavior.

Tests cover all seven page scripts, overview action handoffs, API wire schemas and
optional fields, offline/422/timeout/malformed responses, safe citations, repeated
clarifications, failed-resume retry, draft editing, New Chat, session isolation,
blocked showcase requests, goal math, portfolio reconciliation/comparisons,
watchlists, empty metadata, theme contrast, and the HTTP-only import boundary.

## Running-server integration results

Both services were already running locally on ports 8000 and 8501.

| Check | Result |
| --- | --- |
| API `/health` | HTTP 200, status ok |
| Running OpenAPI paths/components vs supplied snapshot | Exact match |
| Streamlit health and all seven page URLs | HTTP 200 |
| Finance QA | finance_qa_node, answered, 2 sources |
| Portfolio customer C001 | portfolio_analysis_node, answered, 1 calculation |
| Market AAPL quote | market_analysis_node, answered; text-only quote response |
| Goal feasibility | goals_node, answered, 1 calculation |
| News synthesis | news_synthesizer_node, answered, 2 sources |
| Tax education | tax_education_node, answered, 5 sources |
| Underspecified savings goal → clarification → resume | Answered, 1 calculation |
| Invalid `/query` and `/query/resume` inputs | Both HTTP 422 |

The first live run lost Tax and final Goals connections while source files were being edited;
health recovered, and both passed on a source-stable rerun. Development reloads
are the likely cause; this was not proven from backend process logs. The smoke tool deliberately
does not automatically retry queries. Use FastAPI without `--reload` for interviews.
No personal customer data or credentials were used in these synthetic test questions.
These live results are observations, not guarantees of future provider availability.

## Accessibility and responsive QA

Native labeled inputs, buttons, navigation, skip-to-content link, retained focus
controls, text statuses, chart alternatives, and educational disclaimer are present.
Theme contrast tests require at least 4.5:1 for main/sidebar text, links, and primary
button text. Manual Chrome checks inspected all seven page views and the real API connection
indicator. Keyboard Tab/Enter navigation to Goals worked; goal inputs and metric
cards remained readable at 200% zoom (a narrower effective viewport). Earlier
checks also verified native sidebar collapse and card wrapping. Browser zoom was
restored afterward. All page execution and interactive paths
are covered by AppTest. This is not a full WCAG or screen-reader certification;
Plotly interaction and wide holdings tables need broader assistive-device testing.

## MVP boundaries and remaining work

1. Introduce versioned typed market time-series, provider timestamp, cache/delay, and
   news publication metadata. Do not infer these from generated prose.
2. Formalize portfolio calculation schemas and historical-return/cash-flow semantics.
   Current supported backend snapshots are synthetic portfolios with frozen prices;
   UI samples are separate fictional examples selected explicitly.
3. Add idempotency/cancellation and server-supported conversation context. A request
   timeout has unknown completion; ordinary questions currently receive no prior
   chat messages. New Chat clears only browser-session UI state, not checkpoints.
4. Expand reproducible automated viewport/keyboard/screen-reader browser coverage.
5. Address existing backend lint/type/deprecation debt before production hardening.

No login, new role enforcement, durable UI chat history, direct agent invocation, or
new API route was introduced. Existing six agents, LangGraph workflows, FastAPI,
RAG, stores, and simulated backend portfolio access remain intact.
