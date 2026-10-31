# Router architecture: ordered execution specification

Read after the [system sequence](01-system-sequence.md). The router is a single-domain classifier plus deterministic handler selection. It is not a supervisor that repeatedly delegates among specialists.

## 1. Construction

1. FastAPI's cached graph dependency calls `create_intent_classifier` with application settings.
2. Construct a LangChain `ChatOpenAI` client using `OPENAI_MODEL` (source default `gpt-4o-mini`), the configured key, the router timeout, and one client retry. The outer classification deadline also bounds the operation; client retry does not imply an unbounded retry loop.
3. Bind Pydantic `IntentClassification` as structured output with a single supported `intent` value.
4. Load `INTENT_ROUTER_SYSTEM_PROMPT`, version `v3` by source default, or the configured prompt override. Prompt version metadata labels requests; a label alone does not verify deployed content.
5. Register the async classifier within the parent graph's `intent_router`, with the displayed inner run name Classify request. Register a separate Choose specialist agent transition for execution routing.

Source anchors: [classifier](../../src/finassist/workflow/classifier.py), [prompt](../../src/finassist/prompts/system_prompts.py), [graph assembly](../../src/finassist/workflow/builder.py).

## 2. Per-request ordering

1. Receive a request only after API input redaction and the graph input guard allows it. A blocked request does not invoke the router model.
2. Check deterministic standalone-greeting detection. If matched, return the welcome answer, mark the greeting, and route to the output guard without an LLM classification call.
3. Otherwise send the classifier system prompt and redacted canonical question to the structured model. The classifier is told to classify, not answer.
4. Apply the classification deadline, then validate the structured enum. Log the intent or a safe failure category with elapsed time.
5. Compute the handler from intent plus the limited generic-query heuristics. Store `intent`, `classified_intent`, `executed_handler`, and `standalone_greeting=False`.
6. Choose specialist agent follows `executed_handler`. Finance/Tax may then enter their semantic cache wrappers; the router itself has no semantic classification cache.
7. The specialist owns domain-specific ambiguity, evidence, tools, and clarification. Domain identification does not establish that a question has enough inputs to answer.
8. On completion, specialist output returns to the shared output guard. There is no automatic specialist-to-router-to-another-specialist cycle.
9. On a real clarification resume, LangGraph continues the checkpointed branch instead of classifying the reply as an independent question.

## 3. Classification policy

1. Classify actual task meaning rather than incidental words or UI page names. The UI may generate wrappers, so trace the exact API question when diagnosing a decision.
2. Tax covers US tax education, filing, deductions, credits, and retirement-account eligibility/contribution/withdrawal/conversion rules.
3. Finance Q&A covers general investment education, instruments, risk, diversification concepts, simple/compound interest, and general causal explanations of rates/inflation effects.
4. Portfolio covers personal/manager holdings, value, allocation, concentration, unrealized performance, manager comparisons, screening, and customer counts. Assignment/directory phrasing can still be misclassified; there is no hard-coded guaranteed mapping for every paraphrase.
5. Market covers current quotes, symbol resolution, history, period returns, and cross-asset price performance. Company names alone do not imply a personal holdings request.
6. Goals covers savings targets, contribution planning, future value, time to target, and deadline feasibility.
7. News covers current/recent events, financial/technology/industry reports, earnings and dated macro/market movements. Educational effects belong to Finance; observed price/history requests belong to Market.
8. `clarify` means the domain itself cannot be identified. If the domain is known but ticker, filing status, or other inputs are missing, the prompt instructs routing to that specialist.
9. `other` means outside supported domain categories. Do not describe `other` as a definitive safety assessment; safety guards execute separately before classification.
10. Multiple domain mentions do not create parallel agent execution. The enum supports one selected intent. Complex cross-domain tasks may need separate questions.

## 4. Deterministic routing after classification

1. `finance_qa` selects `finance_qa_node`; `tax` selects `tax_education_node`; `news` selects `news_synthesizer_node`.
2. `portfolio` selects `portfolio_analysis_node`, except configured generic-phrase matches can select `clarify_node` instead.
3. `market` selects `market_analysis_node`, except certain generic questions select `clarify_node`. The generic-query check extracts the user question from the known Market UI wrapper before matching. When market quotes are disabled, route to Market so its disabled-feature answer can be returned.
4. `goals` normally selects `goals_node`; a generic-query helper exists, though the current Goals phrase list is empty.
5. `clarify` selects the generic `clarify_node`; unmatched intents select `unsupported_node`.
6. Generic `clarify_node` emits a prompt asking for more details as a completed answer. Unlike specialist clarification tools, it does not create a resumable interrupt/thread. Draw these as distinct flows.
7. Deterministic selection uses scope heuristics only for those generic branches; it does not validate portfolio ownership or invoke market/tax tools.

## 5. Failure and observability sequence

1. Router timeout becomes `ClassificationTimeoutError`; provider/API failure becomes `ClassificationProviderError`; invalid structured output becomes `ClassificationOutputError`.
2. FastAPI maps those typed classification errors to HTTP 504/503/502 through the query/resume safe-error boundary. Other caught execution errors can still return HTTP 200 with a domain-error status. See the current API contract and evaluation results.
3. Model call and graph spans expose classification timing and available usage. Prompt version, environment, source, and handler labels help compare decisions.
4. A request rejected by regex, moderation, or semantic injection screening may be labeled `other`/`unsupported_node` without the classifier ever running. Read the path, not only the final labels.
5. Tests and router evaluation datasets compare classification outcomes and latency. A scripted routing test verifies mappings/contracts, not universal model consistency. Live repeated evaluation is needed to assess probabilistic routing accuracy.

Source anchors: [routing rules and greeting handling](../../src/finassist/workflow/builder.py), [greetings](../../src/finassist/workflow/greetings.py), [API exception boundary](../../src/finassist/api/query.py), [global handlers](../../src/finassist/main.py), [router evaluation](../../src/finassist/evaluation/router/evaluation.py).
