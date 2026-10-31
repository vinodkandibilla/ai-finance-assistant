# Portfolio Analysis: understanding demonstration holdings

Maya opens Portfolio and sees synthetic customer accounts. The dashboard resembles an account tool, but its records and frozen reference prices belong to the demonstration store. Portfolio Analysis explains those holdings; it does not connect to her brokerage.

## What the agent does

Scoped tools resolve customers, retrieve portfolios and investment profiles, screen holdings, and compare a manager’s customers. Deterministic calculations produce value, allocations, concentration, and unrealized gain/loss where cost basis is complete. Rankings include concentration, diversification, and unrealized loss. Holding HHI measures concentration in holding weights, not overall suitability or all investment risks.

## From simple to hard

| Level | Example question | Expected processing |
| --- | --- | --- |
| Simple | “Show the selected customer’s portfolio value and largest holding.” | Resolve a customer in supplied scope and report calculated holdings values. |
| Medium | “For this customer, compare asset-type and sector allocations and explain concentration.” | Use calculated allocations and concentration metrics without inventing a suitability assessment. |
| Hard | “Compare my assigned customers by concentration and identify portfolios with unrealized losses.” | Use scoped comparison/screening tools, state ranking metrics, and disclose excluded or incomplete portfolios. |

These are illustrative supported paths, conditional on available data and evidence; they are not captured live responses.

## When it asks for clarification

Missing customer identifiers or multiple name matches need clarification. Use identifiers from the demonstration directory or a selected customer; “my” does not create an authenticated identity. Tools restrict customer and manager access to supplied simulated scope.

## What it declines or cannot perform

Out-of-scope customer/manager access is denied. Selling holdings, changing accounts, and retrieving unconnected real holdings cannot be performed. Existing simulated scope checks do not provide production authentication or authorization, which remain deferred.

## When the answer remains unknown

Empty holdings, invalid valuation data, and zero portfolio value can yield insufficient results. Missing cost basis prevents complete unrealized-performance reporting and can exclude a customer from loss rankings. Excluded accounts must not be treated as zero losses.

## How to read the result

Check price dates, synthetic-data labels, customer/manager scope, and incomplete data notices. Unrealized gain/loss from frozen prices is not a historical return series. Holding count alone does not establish diversification. The dashboard provides customer search, Summary, Positions, and Allocation tabs; chat can show comparison tables.

Implementation: [agent](../../src/finassist/agents/portfolio_analysis/agent.py) and [tools or validation](../../src/finassist/agents/portfolio_analysis/tools.py). Continue with [the agent journey](README.md) and [UI walkthrough](../ui-guide.md).
