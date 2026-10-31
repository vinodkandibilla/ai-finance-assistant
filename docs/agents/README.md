# A journey through FinPilot’s specialists

Imagine Maya opening FinPilot to understand an ETF. Later she studies taxes, checks prices, explores demonstration holdings, plans a savings goal, and reads financial news. Each question needs a different source of truth. FinPilot routes it to one of six specialists.

These pages describe the current implementation. Examples illustrate expected processing, not recorded live responses or guarantees of available evidence. Simple, medium, and hard mean increasing inputs, tools, or evidence needs; they are documentation categories, not runtime complexity classifications.

## How a question travels

Maya submits a question in **Ask FinPilot**. The API validates it; input guardrails screen it and redact PII before model routing. The router chooses exactly one domain. The specialist retrieves evidence or calls calculators/provider tools, then the output guardrail sanitizes and validates the result. The UI displays the answer, status, sources, and available calculation details.

A workspace provides controls and suggested questions; the actual question determines routing. “What is diversification?” is Finance Q&A. “How concentrated are my holdings?” is Portfolio. “Compare AAPL and MSFT over the past month” is Market. “Why did technology stocks fall today?” is News. Retirement-account eligibility and contribution rules belong to Tax.

The router is not a general planner that coordinates all six agents for one request. Split cross-domain work into separate questions. Finance, Tax, and Goals explicitly reject changed clarification tasks outside their scope and request a new question rather than automatically transferring them.

## Understand the outcome

| Outcome family | Meaning | What Maya can do |
| --- | --- | --- |
| Answered | Evidence or successful tools support the result. | Read assumptions, dates, and sources. |
| Clarification | Essential information is missing or the task is unclear. | Reply on a real paused thread; router-level clarification without a thread requires a new complete question. |
| Insufficient / unknown | The domain is relevant, but supported evidence or calculation coverage is incomplete. | Narrow the question or supply missing information. |
| Out of scope | This specialist cannot perform the requested kind of work. | Submit a new supported question. |
| Blocked / denied | Shared safety or portfolio scope prevents the request. | Use a permitted request within scope. |
| Error | A tool, provider, deadline, or execution failed. | Retry when appropriate; do not interpret failure as a financial conclusion. |

These are explanatory families, not one universal status enum. Market can use `clarify`; other specialists have their own contracts. Missing evidence is different from a safety refusal. Missing provider data is different from a zero value.

## Meet the specialists

| Specialist | Source of authority | Story |
| --- | --- | --- |
| Finance Q&A | Handbook and interest calculators | [Learn investing concepts](finance-qa.md) |
| Tax Education | Bundled US tax evidence and supported federal calculator | [Keep tax years and assumptions explicit](tax-education.md) |
| Market | Twelve Data observations and market calculation tools | [Read prices in context](market-analysis.md) |
| Portfolio | Scoped synthetic holdings and deterministic calculations | [Explore demonstration accounts](portfolio-analysis.md) |
| Goals | Deterministic savings calculators | [Turn a target into a scenario](goals.md) |
| News | Retrieved articles and source-linked claims | [Follow reports to their evidence](news.md) |

Continue with the [UI walkthrough](../ui-guide.md). Shared protections are documented in [guardrails](../architecture.md#guardrails); deferred production features appear in [MVP scope](../mvp-scope.md).
