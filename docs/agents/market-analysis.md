# Market Analysis: prices with dates attached

Maya now asks for AAPL’s latest available price. Market Analysis resolves uncertain symbols and requests provider data. An observation belongs to a listing, currency, exchange, and timestamp; it is not a guarantee of an executable real-time price.

## What the agent does

Twelve Data tools supply quotes, symbol searches, and historical observations. Deterministic tools calculate price growth and historical extrema. Python validates timeframes; generated prose must not invent values or substitute its own arithmetic. Quotes cache for 30 minutes; history and symbol searches cache for 24 hours.

## From simple to hard

| Level | Example question | Expected processing |
| --- | --- | --- |
| Simple | “What is the latest available price of AAPL?” | Fetch a quote and show available provider fields and observation time. |
| Medium | “Show AAPL daily prices over the past month.” | Resolve the time window and return dated history and structured chart data when available. |
| Hard | “Compare AAPL, MSFT, and NVDA price growth from 2026-09-01 to 2026-09-30.” | Use the same validated window and price-growth tool for all assets; disclose unavailable observations. |

These are illustrative supported paths, conditional on available data and evidence; they are not captured live responses.

## When it asks for clarification

Ambiguous company/listing matches need clarification. Fiscal quarters, differing fiscal periods, or unspecified comparison windows can require common explicit dates. Comparisons support at most four assets; a fifth cannot silently be dropped. Requests allow at most 100 records: narrow the period or coarsen the interval instead of paginating around the limit. Extrema use daily observations across at most 100 calendar days; maximum defaults to highs and minimum to lows unless open/close is requested.

## What it declines or cannot perform

Guaranteed future prices and trade recommendations are outside its objective data role. Current-event explanations belong to News; general investing concepts belong to Finance Q&A. Tools cannot execute orders.

## When the answer remains unknown

Unavailable fields remain unavailable, not guessed or zero. Failed tools cannot justify fabricated prices. Daily quote change is not period return, and an incomplete current bar is not a completed trading interval.

## How to read the result

Check listing, exchange, currency, provider timestamp/timezone, interval, common window, and incomplete-bar notes. The stock-oriented Market UI supports NASDAQ/USD views, two-stock comparison, and session watchlists; it exposes a narrower surface than the agent’s broader instrument query capabilities.

Implementation: [agent](../../src/finassist/agents/market_analysis/agent.py) and [tools or validation](../../src/finassist/agents/market_analysis/timeframe.py). Continue with [the agent journey](README.md) and [UI walkthrough](../ui-guide.md).
