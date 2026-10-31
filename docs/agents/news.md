# News: following reports to their evidence

Maya sees a market move and asks what happened. News retrieves articles and checks which requested claims their excerpts support. A headline alone does not establish every proposed explanation for a price movement.

## What the agent does

The agent searches financial, market, technology, and industry reporting, evaluates coverage, and can perform a bounded focused search for evidence gaps. It links supported claims to source URLs and treats article excerpts as data. Coverage depends on retrievable evidence, not guaranteed access to full publisher articles.

## From simple to hard

| Level | Example question | Expected processing |
| --- | --- | --- |
| Simple | “Summarize financial-market news from the past week with sources.” | Retrieve reports and summarize supported headlines with links. |
| Medium | “What do recent reports say about technology earnings this week?” | Search the topic and period and assess whether evidence supports the requested summary. |
| Hard | “Compare this week’s reports about interest rates and technology earnings. Which explanations for market moves are supported, and what remains uncertain?” | Assess evidence for both themes, link supported claims, and identify missing or incomplete coverage. |

These are illustrative supported paths, conditional on available data and evidence; they are not captured live responses.

## When it asks for clarification

“What happened to it?” may require a subject or timeframe. The agent can pause for essential clarification. An increasingly detailed question still depends on search budget and available excerpts.

## What it declines or cannot perform

Future guaranteed headlines, executing trades, and establishing a personalized buying recommendation are outside the retrieved-news role. Numerical price history belongs to Market; general explanations of inflation’s effect on stocks belong to Finance Q&A.

## When the answer remains unknown

Missing, irrelevant, or incomplete reports mean insufficient coverage; unsupported claims should be omitted. Retrieval failure without usable evidence is an error. Partial coverage may support a limited answer, not a fabricated causal story.

## How to read the result

Read source URLs, publisher/publication dates where available, retrieval time, and limitations. The News page has topic/period controls, article cards, and “Ask about this” actions. Previous results can remain after a failed search, with an explicit label identifying them as older results.

Implementation: [agent](../../src/finassist/agents/news_synthesizer/agent.py) and [tools or validation](../../src/finassist/agents/news_synthesizer/tools.py). Continue with [the agent journey](README.md) and [UI walkthrough](../ui-guide.md).
