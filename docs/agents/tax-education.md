# Tax Education: keeping the year visible

Maya next asks how a deduction differs from a credit. Tax Education retrieves a US tax explanation without demanding personal information for a definition. When she asks for her bracket or a calculated amount, the required inputs become more specific.

## What the agent does

The agent retrieves and evaluates bundled US tax evidence, with bounded expansion and targeted recovery. Its default corpus scope is tax year 2026, generally filed in 2027. Explicit years must be preserved. Retirement-account tax rules belong here when supported. The deterministic calculator supports 2026 federal ordinary income tax in USD, before credits and payments; it does not prepare a complete return.

## From simple to hard

| Level | Example question | Expected processing |
| --- | --- | --- |
| Simple | “What is a federal tax deduction?” | Retrieve a concise cited definition without asking for income or filing status. |
| Medium | “For 2026, explain standard versus itemized deductions.” | Preserve the year and support both sides of the comparison. |
| Hard | “For 2026, calculate federal ordinary income tax on $130,000 of taxable ordinary income, filing single. Show bracket slices and distinguish marginal and effective rates.” | Use the calculator for amounts and retrieval for explanations. State the income basis and exclusions. |

These are illustrative supported paths, conditional on available data and evidence; they are not captured live responses.

## When it asks for clarification

“What is my bracket?” requires income and filing status. Salary, gross income, wages, and take-home pay must not silently become taxable ordinary income. The agent requests the needed taxable-income basis. Income already described as taxable must not have deductions subtracted again. Credits, payments, preferential gains/dividends, AMT, and other unsupported features require agreement to a simplified calculation before omission.

## What it declines or cannot perform

Filing a return, moving IRA funds, executing trades, and account actions are outside this educational role. Non-tax finance, quotes, news, portfolio analysis, and goal projections require a new question when introduced during clarification.

## When the answer remains unknown

An unsupported explicit year must not be replaced with 2026. Missing year-specific or jurisdiction-specific evidence means insufficient support. The ordinary-income calculator cannot produce a complete return or infer state/foreign rules from federal material. Tool failure without usable evidence means error.

## How to read the result

Check tax year, filing status, income basis, marginal versus effective rates, citations, and exclusions. The Tax Education workspace supplies learning prompts; chat displays supported results and available calculation payloads.

Implementation: [agent](../../src/finassist/agents/tax_education/graph.py) and [tools or validation](../../src/finassist/agents/tax_education/tax_calculation.py). Continue with [the agent journey](README.md) and [UI walkthrough](../ui-guide.md).
