# Architecture

FinPilot uses FastAPI as the service boundary, LangGraph for orchestration, specialist agents for domain behavior, Qdrant for retrieval, and Streamlit as the client.

The graph is:

```text
input_guardrail_node → intent_router → specialist agent → output_guardrail_node → END
```

The API performs request validation, protects raw PII before graph entry, supplies simulated portfolio context, and retains a last-resort exception boundary. Agents remain responsible for domain evidence, citations, calculations, and tool semantics.

The design favors explicit state transitions, bounded tool calls, checkpointed clarification, provider data provenance, and LangSmith traceability. See [guardrails](guardrails.md) and [observability](observability.md).
