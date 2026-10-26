# Observability

LangSmith traces the LangGraph run tree and its child model, tool, retrieval, and guardrail runs. The parent view can show aggregate latency, token usage, and cost when child usage metadata is available; detailed usage remains on child spans.

Guardrail spans include `regex_guard`, `moderation_guard`, `injection_guard`, and `output_guard`. Prompt version, environment, source, and safe guardrail outcomes are attached as metadata. The OpenAI Usage dashboard remains the billing source of truth.
