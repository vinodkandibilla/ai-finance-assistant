# Finance Q&A failure checks: MVP

Reuse the production runtime and existing offline tests; no separate evaluator.

## Covered

- Empty search: an attempted unsupported answer becomes `insufficient`, with no
  fabricated answer text or citations (`tests/test_finance_autonomous_agent.py`).
- Retriever exception or provider timeout: an attempted unsupported answer becomes
  `error`, with safe error information and no invented citations (same suite).
- Agent deadline during actual retrieval: cancels the pending call and returns
  `error` with `agent_timeout` (`tests/test_agent_deadlines.py`).
- Existing recovery-tool error handling (`tests/test_finance_evidence_recovery.py`).
- Existing reranker failure prevents unranked evidence leaking through
  (`tests/test_rag_tool_pipeline.py`).
- Existing missing-index/empty-corpus retrieval behavior (`tests/test_finance_qa.py`).
- Existing full-agent deadline, safe timeout logging and settings validation
  (`tests/test_agent_deadlines.py`).

```bash
pytest -q tests/test_finance_autonomous_agent.py tests/test_agent_deadlines.py \
  tests/test_finance_evidence_recovery.py tests/test_rag_tool_pipeline.py \
  tests/test_finance_rag_tool.py
```

Tests inject failures and scripted model responses, exercising actual runtime
handling without provider calls. They do not measure live provider reliability,
retry effectiveness, operational latency or the model's decision to recover.
No production behavior, new retry policy or live failure simulation is introduced.
Provider exception text is excluded from returned answers/errors by these checks;
this does not certify every logging path. Add operational checks only as needed.
