# Exploring FinPilot: a UI walkthrough

Maya opens FinPilot to learn, explore data, and try a savings scenario. The interface keeps a workspace beside **Ask FinPilot**, so she can inspect a chart or learning prompt while reading an answer. This guide follows her through the implemented Streamlit interface. It describes source behavior, not a recorded browser session or a guarantee that external providers are available.

## Getting oriented

The sidebar shows **FinPilot AI**, “Learn. Explore. Plan.”, and seven destinations: Overview, Finance Q&A, Portfolio, Market, Goals, News, and Tax Education. **Check API connection** runs a health check and shows a success message with API version or an error. A successful health check does not establish that every market/news provider or retrieval source works.

The main area has a page title and description, a left content pane, and a right assistant pane. Initial widths are approximately 62% content and 38% chat, with a resizable divider. Workspace content and chat history have scrollable regions. Educational context appears with workspace content. The sidebar labels the experience **Educational workspace · Session only**.

Navigation preserves the shared conversation during the Streamlit session. It does not provide durable saved chat, brokerage access, or authenticated customer accounts. The actual submitted question determines the routed specialist; selecting Market does not force an educational question to use market tools.

## First stop: Overview

Maya sees **What would you like to explore?** and six cards: Finance Education, Portfolio Analysis, Market Intelligence, Goal Planning, Financial News, and Tax Education. Each **Explore** action opens the corresponding workspace with a suggested question for the chat handoff.

Below, **A glimpse of your workspace** contains a demonstration market preview and a demonstration portfolio preview. Fictional indexes and the example $100,000 portfolio with a 60/30/10 allocation illustrate the interface. They are not live observations or connected holdings. Following a preview opens the relevant workspace with an educational question; it does not turn those fictional preview numbers into provider data.

## Learning in Finance Q&A and Tax Education

Maya selects **Choose a learning topic**. Finance offers Investing Basics, Banking & Interest, Bonds & Funds, and Risk & Diversification. Tax offers its Tax Education topic. Each topic includes an introduction and six suggested questions that grow from definitions to comparisons and multipart examples.

Clicking a learning question fills and focuses the chat draft. Maya can edit it before sending; selecting the suggestion is not itself a completed agent response. The source groups prompts into Simple, Medium, Semi-medium, and Complex; the agent guides use simple/medium/hard for readability. Neither represents a runtime guarantee of latency or answerability.

A pending clarification or failed chat request disables learning suggestions. Maya must finish the reply, resolve the failed request, or choose New Chat. Explanations come from the backend, not prewritten answers inside the learning cards. See [Finance Q&A](agents/finance-qa.md) and [Tax Education](agents/tax-education.md) for evidence and calculator boundaries.

## Inspecting Portfolio

Maya opens **Portfolio analytics** and its **Portfolio Data** explanation. The workspace loads demonstration portfolio data through the API. The manager selector allows switching only when supplied context permits it; manager changes clear dependent selections. A pending conversation can constrain changes to avoid mixing contexts.

For a manager view, **Your customers** provides customer count, total displayed portfolio value, and search. The directory shows customer information and portfolio summaries; selecting a customer opens the detailed snapshot. A customer view can instead show **Your portfolio**. No matches produces an explicit empty state rather than fabricated customers.

The snapshot identifies customer, manager, and price dates and offers three tabs:

| Tab | What Maya sees | How to interpret it |
| --- | --- | --- |
| Summary | Portfolio metrics, largest holding, cost basis and unrealized performance when complete, and concentration context | Frozen-price unrealized gains/losses are not a historical return series. |
| Positions | Symbol, asset type, sector, market value, allocation percentage, unrealized gain/loss, and price-as-of date | Each row is a synthetic position; incomplete cost basis limits performance reporting. |
| Allocation | Allocation visuals, holding weights, HHI and effective holding count where available | Concentration measures describe these weights; they do not establish suitability or eliminate market risk. |

Unavailable portfolios and incomplete performance receive messages. The dashboard does not replace missing values with a claim of zero loss. Chat responses with portfolio snapshots can show a comparison table and excluded-portfolio notices. Scope checks operate on simulated context; production authentication and authorization remain deferred. See [Portfolio Analysis](agents/portfolio-analysis.md) and the [portfolio API](api/portfolio-dashboard.md).

## Exploring Market

The **Market overview** is a NASDAQ/USD stock-oriented interface. Provider quotes may be delayed or cached. Maya chooses **Individual stock** or **Compare stocks**.

In the individual view she enters a symbol, initially AAPL, and presses Enter. **Daily** shows a one-month window; **Weekly** shows three months. The interface validates symbol format and asks the backend for market data. Returned quote metrics can include latest price, open, high, low, volume, and previous close. Missing fields are not invented. The quote observation is displayed separately from history.

Validated OHLC bars produce a candlestick chart. The latest bar’s open/high/low/close appears above it; a dotted latest-quote line appears when a valid quote is supplied. The chart identifies the last bar and warns that the current bar may be incomplete. No valid series produces an informational state; unavailable data produces an error. There is no automatic price polling.

The comparison view accepts two different stock symbols and an interval. It requires at least two shared observation dates. **Actual price (USD)** compares closing prices; **Start = 100** compares relative changes from a common starting value. Hover details include closing price and index, and **Closing prices by date · USD** expands the underlying table. Price-change metrics use the shared observation window and exclude dividends; split/adjustment treatment depends on provider data. Read the period and chart labels before interpreting performance. This two-stock control is narrower than the agent query interface, which supports up to four assets within its limits.

**Watchlist** supports named session lists. The plus control opens list creation; names must be nonempty, unique ignoring case, and at most 40 characters. Maya can add or remove symbols and select a symbol to request its quote. The quote card identifies the observation and can be dismissed. Watchlists are session data, and quotes refresh on request, not continuously.

Some provider-backed controls are disabled while chat awaits clarification or a failed request is unresolved. See [Market Analysis](agents/market-analysis.md) for record limits, timeframes, cache behavior, and broader instrument boundaries.

## Planning in Goals

The page first introduces **Turn a goal into a savings plan** with a goal selector and suggested questions. These guide Maya toward supplying the target, horizon, and other necessary assumptions. The coverage expander explains the projection limits.

Next, **Explore a savings scenario** explicitly labels its output **LOCAL HYPOTHETICAL PROJECTION**. Maya changes:

| Control | UI range or meaning |
| --- | --- |
| Target amount | USD, zero to $1 billion |
| Current savings | USD, zero to $1 billion |
| Monthly contribution | USD, zero to $10 million |
| Time horizon | 1–600 months |
| Assumed nominal annual return | 0–20%; a scenario assumption, not a forecast |

The local calculator updates projected ending balance, required monthly saving, and projected shortfall. A chart separates hypothetical balance, starting savings plus contributions, and the target line. Contributions occur at month end with monthly compounding; taxes, fees, inflation, withdrawals, and downside risk are excluded.

Changing these controls does not invoke the Goals agent. **Ask AI about this scenario** queues a question containing the scenario inputs for the backend. The backend explanation is a separate result with its own calculation details and status. UI limits and backend calculator limits differ; neither permits a guaranteed return. See [Goals](agents/goals.md).

## Reading News

Maya chooses a **Category**: US Markets, Technology, Earnings, Economy, or Banking. **Published within** offers Today, Past week, or Past month. An optional **Search topic** overrides the category with a company, sector, or event. **Search news** retrieves source-backed articles.

Results identify the searched topic, period, and retrieval time. Article cards show a title, publisher/publication information when available, and a source link; the leading card is labeled **LEAD STORY**. **Ask about this** hands an article-focused question to the chat workflow.

No supported articles produces a message inviting a narrower topic or another period. A failed search provides **Retry news search**. Previous successful results can remain visible, explicitly labeled as previous results when the latest search did not replace them. They should not be read as a successful fresh search. Pending chat tasks disable conflicting actions. See [News](agents/news.md) for synthesis and evidence boundaries.

## Having a conversation in Ask FinPilot

The chat composer accepts up to 4,000 characters, but the API limits a new question to 2,000; a clarification reply can contain up to 4,000. Longer new questions are rejected rather than silently truncated. During submission the interface indicates work in progress. An assistant turn can contain:

- A specialist label and, when supplied, classified intent and executed handler.
- Markdown answer text and a copy control.
- A status and an error warning where applicable.
- A clarification prompt with instructions to reply below.
- A **Sources** expander with links or source labels. If no sources were supplied, the interface says so; deterministic calculations may legitimately lack citations.
- A **Calculation details** expander with structured payloads, or a portfolio comparison table for portfolio snapshots.

Visible history and backend memory are different. Ordinary new `/query` requests do not send all previous messages. Maya should include the subject and required inputs in a fresh question rather than assuming “compare those again” will reconstruct earlier context. A pending clarification instead uses a thread and `/query/resume` to continue the checkpointed task with her reply.

For example, “Save $50,000 for college” can trigger a deadline question. “In three years” then resumes that same target. A later independent question should restate its own needed context. An unresolved clarification is not a free-form switch to a different specialist.

If a request fails, **Retry last request** resends the stored request. **Edit failed message** restores its text for editing. A retry is another attempt, so it can incur another provider/model call. Error recovery is distinct from an agent reporting insufficient evidence in a completed response.

The New Chat icon clears the current conversation and draft. When there is content to lose, the UI opens **Start a new chat?**, with **Cancel** and **Start new chat**. Clearing cannot be undone. It clears the conversation, not a durable server-side account history; none is implemented.

## When the interface cannot finish

A health-check error points to backend connectivity/configuration. A provider failure can affect Market or News even if health succeeds. A clarification asks for user input. An insufficient response signals evidence or calculation coverage limits. An out-of-scope response signals the wrong kind of task. A denial signals a scope restriction. These require different next steps; none should be presented as a confident financial answer.

This walkthrough was checked against [app layout](../src/finassist/web_app/app.py), [chat](../src/finassist/web_app/components/chat.py), [learning](../src/finassist/web_app/components/learning.py), [portfolio](../src/finassist/web_app/components/portfolio.py), [market and goal presentations](../src/finassist/web_app/components/showcase.py), and [news browser](../src/finassist/web_app/components/news.py). For the question-processing story, continue with [Meet the specialists](agents/README.md).
