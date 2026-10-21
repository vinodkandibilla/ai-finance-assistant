# Finance Q&A retrieval evaluation

The dataset contains 25 cases: 22 answerable questions and three unsupported
requests. It spans the main in-scope handbook topics, with an explicit case-to-topic
map in `coverage.json`. This is a broad regression set, not an exhaustive list of
possible questions or evidence of 80% real-user coverage.

| Topic area | Selected examples |
| --- | --- |
| Foundations and growth | Budget categories, net worth, inflation, simple/compound interest |
| Saving and banking | Time horizons, APY, fees and FDIC aggregation/securities exclusion |
| Borrowing | APR, interest rate and loan-term tradeoffs |
| Stocks and bonds | P/E caveats and bond interest-rate risk |
| Funds and indexes | NAV, ETF pricing, expense ratios, active/passive management |
| Portfolio concepts | Diversification and risk capacity/tolerance |
| Practical investing | Market versus limit orders |
| Retirement income | Longevity risk |
| Complex products and options | Daily-reset funds, call/put rights and premium multipliers |
| Investor protection | Fraud warning signs |
| Unsupported evidence | Current bank offers, live quotes and private holdings |

Tax-account rules, personalized goal projections and portfolio analysis remain
outside Finance Q&A routing scope. Their appearance as a negative case does not
mean they should route to this agent. General portfolio and retirement-income
concepts are education cases; no personal holdings or account rules are required.
The compact set omits detailed lending programs, CD variants, individual Greeks,
option strategies and other product-specific questions. Add these as regressions
when needed; this set does not certify every handbook subsection.

## Case schema

Each JSONL record has these required fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique stable identifier. |
| `query` | User question. |
| `dense_queries` | Fixed semantic search inputs, initially the original query. |
| `sparse_keywords` | Fixed lexical search inputs, initially the original query. |
| `answerable` | Whether this frozen corpus can answer the question. |
| `tags` | Topic and retrieval scenario labels. |
| `critical` | Whether a regression requires review. |
| `relevance_groups` | Required concepts and alternative supporting passages. Empty for unanswerable cases. |
| `notes` | Optional labeling explanation. |

Each relevance group has a `concept`, `rationale`, and nonempty `passages` list.
Each passage has a `chunk_id`, physical PDF `page_number` (one-based), and
`content_sha256`. Labels use chunk IDs plus corpus/content hashes instead of
path-dependent production `doc_id` values. `corpus.json` freezes the PDF hash and
chunking configuration. Initial labels were authored by the agent after inspecting
handbook passages and should receive human review; listed passages are not an
exhaustive inventory of every acceptable passage.

Search inputs are fixed to isolate retrieval/reranking from LLM query planning.
These records are an input dataset; a retrieval runner and scoring helpers are
not implemented by this step. No live retrieval baseline has been run.

## Planned scoring interpretation

Evaluate raw hybrid candidates and reranked evidence separately. A concept is
covered when at least one of its labeled alternative chunks appears in the top k.
Hit@5 measures any relevant hit, concept recall@5 measures covered concepts divided
by required concepts, and MRR measures the first relevant rank. Multi-concept
questions require all concepts for a complete-evidence pass; matching one is not
a full pass. Treat unlisted passages as unjudged and review them before declaring
an apparent miss incorrect.

Negative cases are excluded from positive-hit/recall/MRR denominators. They do
not require the retriever to return zero results: generic banking content is
expected to match. They check whether supporting evidence for the specific request
exists; abstention belongs to a subsequent agent/evidence-sufficiency eval.

An 80% hit rate on the 22 answerable cases requires at least 18 hits (81.82%).
That measures this dataset's retrieval performance, not 80% user-query coverage.
No accuracy target is established before an actual baseline.

## Offline validation

```bash
pytest -q tests/test_finance_retrieval_dataset.py
```

This checks that labels still resolve to the frozen corpus and passage content.
It uses local PDF extraction only; it does not call embeddings, Qdrant, an LLM,
or a reranker. Corpus or chunking changes require re-reviewing labels.
