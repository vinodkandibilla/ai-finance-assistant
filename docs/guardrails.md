# Guardrails

Input guardrails run in `input_guardrail_node`: basic request limits, Presidio PII redaction, deterministic regex checks, OpenAI Moderation, and an optional semantic injection classifier. Raw PII is redacted before external model calls.

Output guardrails run in `output_guardrail_node`: final-answer PII sanitization, empty-answer fallback, and common response validation. Domain agents retain evidence, citation, market-provider, portfolio-calculation, and tax-specific checks.

Guardrail runs record safe decisions and entity types only. They do not record raw queries or sensitive payloads. Portfolio MVP responses preserve synthetic customer IDs and names; production authorization is deferred.
