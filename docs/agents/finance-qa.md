# Finance Q&A: learning before acting

Maya begins with “What is an ETF?” She needs an explanation, rather than a price or an account action. Finance Q&A searches the finance handbook, checks support for the requested facts, and cites the passages it uses. A definition normally receives a short explanation; a multipart comparison needs evidence for each requested part.

## What the agent does

Hybrid dense and lexical retrieval finds handbook passages. The agent assesses coverage and can expand or target searches within a bounded budget. Simple- and compound-interest tools handle single-principal arithmetic. Pure calculations need no retrieval; accompanying educational explanations do. Missing handbook support cannot be replaced by an invented fact.

## From simple to hard

| Level | Example question | Expected processing |
| --- | --- | --- |
| Simple | “What is a stock?” | Retrieve a supported definition and provide a concise cited explanation. |
| Medium | “Compare stocks and bonds by ownership, income, and main risks.” | Find evidence for each requested comparison dimension; identify unsupported portions. |
| Hard | “Compare simple and compound interest on $1,000 at 5% annually for 3 years, with annual compounding, no fees or deposits. Show interest and final balances and explain the difference.” | Call both calculators with shared inputs and retrieve support for the explanation. Keep interest distinct from ending balance. |

These are illustrative supported paths, conditional on available data and evidence; they are not captured live responses.

## When it asks for clarification

“Calculate my interest” requires essential missing inputs. Ambiguous rates, durations, methods, or contradictory inputs need clarification. Explicit deposits, withdrawals, taxes, fees, or other unsupported features cannot silently disappear: the agent seeks agreement before omitting them. Failed arithmetic must not be replaced with mental calculations.

## What it declines or cannot perform

“Buy the best ETF for my account” is outside its educational role. Tax rules, live prices, news, personalized holdings, savings plans, trades, and account actions belong outside this specialist. A changed clarification task outside scope requires a new question.

## When the answer remains unknown

A niche concept absent from the handbook remains insufficient after the allowed evidence recovery; it is not automatically out of scope. A retrieval failure without usable evidence is an error. More complex wording cannot create missing evidence.

## How to read the result

Read the citations and calculator inputs/defaults. The Finance page offers learning topics and editable question suggestions; Sources and Calculation details appear in chat when supplied.

Implementation: [agent](../../src/finassist/agents/finance_qa/graph.py) and [tools or validation](../../src/finassist/agents/finance_qa/tools.py). Continue with [the agent journey](README.md) and [UI walkthrough](../ui-guide.md).
