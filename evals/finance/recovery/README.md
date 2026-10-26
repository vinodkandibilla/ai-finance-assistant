# Finance recovery decision cases

Two paired fixtures reproduce the stock-definition failure while retaining justified
recovery. Both ask the same question; only the baseline evidence changes. These
are controlled evidence fixtures, not assertions that live Qdrant returns these
exact passages.

For a live decision check, supply the original question and the fixture passages
as an initial `finance_rag_query` tool result, using the production finance prompt
and tools. Compare the next model-selected tool with `expected_next_tools`.
For recovery, check that `missing_aspect` explains the actual unsupported fact.
For completion, check that the answer defines ownership and cites page 30.

These cases are not wired into an automated live runner and no live model calls
were made for this change. Scripted integration tests validate the tool protocol,
validation, budgets and completion; they do not measure model decision accuracy.
