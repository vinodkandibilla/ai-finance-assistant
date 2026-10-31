# Architecture

FinPilot uses Streamlit as the client, FastAPI as the service boundary, LangGraph for routing and specialist execution, Qdrant for Finance/Tax retrieval, and domain-specific tools and providers. Each request selects one specialist; the router does not repeatedly delegate among all six agents.

![System architecture](architecture/assets/finpilot-architecture-v2.png)

[System SVG](architecture/assets/finpilot-architecture-v2.svg) · [Routing and HITL PNG](architecture/assets/finpilot-request-routing-and-hitl.png) · [Routing and HITL SVG](architecture/assets/finpilot-request-routing-and-hitl.svg)

## Request flow

1. Streamlit sends a question through FastAPI. The API validates fields, redacts input and supplies simulated portfolio identity.
2. The input guard applies PII redaction, regex checks, configured moderation and semantic injection screening.
3. The intent router classifies the task and selects a handler. Standalone greetings bypass model classification. Router clarification returns normally; unsupported requests select `other`.
4. The specialist retrieves evidence, calls a provider or runs deterministic Python calculations. Finance/Tax can first check eligible semantic answer caches.
5. A specialist can interrupt for essential missing information. SQLite checkpoints preserve execution; `/query/resume` continues the saved branch using the same thread and identity scope.
6. Output redaction and API response validation precede UI display. Classification failures return 502/503/504; completed domain outcomes retain their own status semantics.

The parent graph state carries the current task and result fields. Specialist messages retain tool calls, results and clarification replies. UI history keeps messages visible but is not automatically sent with each new question. Checkpoints support task continuation; general long-term user memory is not implemented.

## Guardrails

API request validation enforces field lengths and types. The graph input guard then runs Presidio PII redaction, deterministic regex checks, OpenAI moderation and injection classification when configured. Moderation and injection calls currently run sequentially; there are no separate timing spans for every subcheck. A regex, moderation or injection block skips normal classification and specialist execution.

The graph output guard redacts answer text. The API also sanitizes the public answer, handles empty-answer fallback and validates the response model. These checks do not establish financial correctness. Domain agents own evidence/citation and calculation checks. A valid paused clarification must retain its question and thread rather than becoming an empty-answer fallback.

Do not describe this as a general output toxicity/competitor pipeline. Structured calculations and citations are not recursively PII-redacted by the answer-text check. Trace previews redact text, but that does not prove every child tool/message payload is sanitized. Full initial-guard parity on resume remains deferred.

Portfolio tools enforce scope against application-supplied simulated identity. Actual authenticated accounts and production identity/session controls remain outside the MVP. See [MVP scope](mvp-scope.md).

## Components and reader guides

- [MVP overview](finpilot-mvp-overview.md) and [Word edition](finpilot-mvp-overview.docx): illustrated walkthrough for a presentation or stakeholder review.
- [Specialist guides](agents/README.md): supported questions, tools and evidence boundaries.
- [UI walkthrough](ui-guide.md): page workflows and recovery behavior.
- [API and local operations](api.md): provider setup, configuration and storage lifecycle.
- [Evaluation and observability](evaluation.md): datasets, measured results, traces and limitations.
- [Cost controls](cost-optimization.md): current bounds, caching and deferred tuning.

## Diagram-ready execution specifications

These source-linked sequences provide implementation depth beyond the overview:

1. [System, UI, human interaction, stores and caches](architecture/01-system-sequence.md)
2. [Router construction and execution](architecture/02-router-sequence.md)
3. [Internal sequences for all six specialists](architecture/03-agent-sequences.md)

The [dated architecture review](architecture/04-review.md) records the original review method and known risks. Current settings and the latest evaluation findings take precedence over historical counts or limits.
