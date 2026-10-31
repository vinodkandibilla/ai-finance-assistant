from typing import Any, Literal

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finassist.agents.common.answer_cache import SemanticAnswerCache
from finassist.agents.common.clarification import build_clarification_graph
from finassist.agents.finance_qa import FinanceQaAgent
from finassist.agents.goal_planning import GoalPlanningAgent
from finassist.agents.market_analysis.agent import MarketAnalysisAgent
from finassist.agents.news_synthesizer import NewsSynthesizerAgent
from finassist.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from finassist.agents.tax_education.graph import TaxEducationAgent
from finassist.core.config import get_settings
from finassist.guardrails.injection import classify_injection
from finassist.guardrails.moderation import moderate_text
from finassist.guardrails.pii import redact_pii
from finassist.guardrails.regex import check_regex_guards
from finassist.workflow.classifier import IntentClassifier
from finassist.workflow.state import FinanceAssistantState

type FinanceAssistantGraph = CompiledStateGraph[
    FinanceAssistantState, None, FinanceAssistantState, FinanceAssistantState
]

PORTFOLIO_ANALYSIS_RESPONSE = "Portfolio analysis agent is not configured."
MARKET_QUOTE_UNAVAILABLE_RESPONSE = (
    "I couldn't retrieve a current quote. Check the ticker and try again later."
)
CLARIFY_RESPONSE = (
    "I need a few more details to answer accurately. Please share the specific ticker, "
    "date range, filing status, account type, or topic you want covered."
)


async def _classify_intent(
    state: FinanceAssistantState,
    *,
    classifier: IntentClassifier,
) -> FinanceAssistantState:
    from finassist.workflow.greetings import WELCOME, is_standalone_greeting

    if is_standalone_greeting(state["query"]):
        return {
            "intent": "other",
            "answer": WELCOME,
            "status": "answered",
            "standalone_greeting": True,
        }
    intent = await classifier.classify(state["query"])
    handler = _route_intent({**state, "intent": intent})
    return {
        "intent": intent,
        "classified_intent": intent,
        "executed_handler": handler,
        "standalone_greeting": False,
    }


def _is_generic_query(intent: str | None, query: str | None) -> bool:
    if not query:
        return False

    # Match the user's question, not the Market UI's scope/format instructions.
    if query.startswith("Market scope:") and "\nQuestion: " in query:
        query = query.split("\nQuestion: ", 1)[1]
    normalized = query.strip().lower()
    if not normalized:
        return False

    generic_by_intent: dict[str, tuple[str, ...]] = {
        "market": (
            "what is the stock price",
            "what is the price of the stock",
            "stock price",
            "price of stock",
            "what is the price",
            "market price",
        ),
        "tax": (
            "tax details",
            "irs tax",
            "tax info",
            "what is tax",
            "tax basics",
            "tax help",
        ),
        "goals": (),
    }

    generic_phrases = generic_by_intent.get(intent or "", ())
    is_generic = any(phrase in normalized for phrase in generic_phrases)

    if intent == "market":
        return normalized.rstrip("?.!") in generic_phrases

    return is_generic


def _route_intent(
    state: FinanceAssistantState,
) -> Literal[
    "finance_qa_node",
    "portfolio_analysis_node",
    "market_analysis_node",
    "goals_node",
    "news_synthesizer_node",
    "tax_education_node",
    "clarify_node",
    "unsupported_node",
]:
    intent = state.get("intent")
    if intent == "finance_qa":
        return "finance_qa_node"
    if intent == "portfolio":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "portfolio_analysis_node"
    if intent == "market":
        if not get_settings().market_quotes_enabled:
            return "market_analysis_node"
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "market_analysis_node"
    if intent == "goals":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "goals_node"
    if intent == "news":
        return "news_synthesizer_node"
    if intent == "tax":
        return "tax_education_node"
    if intent == "clarify":
        return "clarify_node"
    return "unsupported_node"


def portfolio_analysis_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": PORTFOLIO_ANALYSIS_RESPONSE}


def finance_qa_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": ""}


def clarify_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": CLARIFY_RESPONSE}


def unsupported_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": "This request is not supported yet."}


def build_graph(
    classifier: IntentClassifier,
    *,
    market_agent: MarketAnalysisAgent | None = None,
    finance_agent: FinanceQaAgent | None = None,
    finance_answer_cache: SemanticAnswerCache | None = None,
    tax_agent: TaxEducationAgent | None = None,
    tax_answer_cache: SemanticAnswerCache | None = None,
    goal_agent: GoalPlanningAgent | None = None,
    news_agent: NewsSynthesizerAgent | None = None,
    portfolio_agent: PortfolioAnalysisAgent | None = None,
    checkpointer: object | None = None,
) -> FinanceAssistantGraph:
    async def classify_node(state: FinanceAssistantState) -> FinanceAssistantState:
        return await _classify_intent(state, classifier=classifier)

    async def input_guardrail_node(state: FinanceAssistantState) -> FinanceAssistantState:
        redacted = redact_pii(state["query"])
        regex_result = check_regex_guards(redacted.text)
        if regex_result.blocked:
            return {
                "query": redacted.text,
                "intent": "other",
                "classified_intent": "other",
                "executed_handler": "unsupported_node",
                "answer": "I can’t help with that request.",
                "status": "blocked",
                "guardrail_blocked": True,
                "pii_entity_types": list(redacted.entities),
            }
        settings = get_settings()
        if settings.openai_api_key and settings.moderation_enabled:
            moderation = await moderate_text(
                redacted.text,
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.moderation_model,
            )
            if moderation.flagged:
                return {
                    "query": redacted.text,
                    "intent": "other",
                    "classified_intent": "other",
                    "executed_handler": "unsupported_node",
                    "answer": "I can’t help with that request.",
                    "status": "blocked",
                    "guardrail_blocked": True,
                    "pii_entity_types": list(redacted.entities),
                }
        if settings.openai_api_key and settings.injection_classifier_enabled:
            injection = await classify_injection(
                redacted.text,
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.injection_classifier_model,
            )
            if injection.blocked:
                return {
                    "query": redacted.text,
                    "intent": "other",
                    "classified_intent": "other",
                    "executed_handler": "unsupported_node",
                    "answer": "I can’t help with that request.",
                    "status": "blocked",
                    "guardrail_blocked": True,
                    "pii_entity_types": list(redacted.entities),
                }
        return {"query": redacted.text, "pii_entity_types": list(redacted.entities)}

    def select_agent_route(state: FinanceAssistantState) -> str:
        if state.get("standalone_greeting"):
            return "output_guardrail_node"
        return state["executed_handler"]

    async def market_analysis_handler(
        state: FinanceAssistantState,
    ) -> FinanceAssistantState:
        if not get_settings().market_quotes_enabled:
            return {"answer": "Market quotes are disabled.", "status": "answered"}
        if market_agent is not None:
            return await market_agent.ainvoke(state)
        return {"answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE, "status": "error"}

    def unavailable_tax_node(_: FinanceAssistantState) -> FinanceAssistantState:
        return {"answer": "Tax education agent is not configured.", "status": "error"}

    def empty_finance_qa_node(_: FinanceAssistantState) -> FinanceAssistantState:
        return {"answer": "", "status": "not_implemented", "citations": []}

    def unavailable_goal_node(_: FinanceAssistantState) -> FinanceAssistantState:
        return {"answer": "Goal planning agent is not configured.", "status": "error"}

    def unavailable_news_node(_: FinanceAssistantState) -> FinanceAssistantState:
        return {"answer": "News synthesizer agent is not configured.", "status": "error"}

    def output_guardrail_node(state: FinanceAssistantState) -> FinanceAssistantState:
        answer = str(state.get("answer") or "")
        redacted = redact_pii(answer)
        if (
            state.get("status") in {None, "answered", "not_implemented"}
            and not redacted.text.strip()
        ):
            return {
                "answer": "I could not produce a verified answer from the available information.",
                "status": "insufficient",
            }
        return {"answer": redacted.text}

    def cached_handler(agent, cache):
        async def handle(state: FinanceAssistantState, config: RunnableConfig) -> dict[str, Any]:
            async def compute():
                return await agent.graph.ainvoke(state, config=config)

            result = await cache.run(
                state["query"],
                compute,
                bypass=bool(config.get("configurable", {}).get("bypass_answer_cache")),
            )
            return {**result, "query": state["query"]}

        return handle

    workflow: StateGraph[
        FinanceAssistantState, None, FinanceAssistantState, FinanceAssistantState
    ] = StateGraph(FinanceAssistantState)
    workflow.add_node("input_guardrail_node", RunnableLambda(input_guardrail_node))
    def add_agent_node(node_name, runnable, agent):
        def select(state, config: RunnableConfig):
            if config.get("metadata", {}).get("finpilot_agent_tags"):
                return runnable.with_config(tags=[f"agent:{agent}"])
            return runnable

        workflow.add_node(node_name, RunnableLambda(select, name=runnable.get_name()))

    add_agent_node(
        "intent_router", RunnableLambda(classify_node, name="Classify request"), "router"
    )
    add_agent_node(
        "finance_qa_node",
        RunnableLambda(
            cached_handler(finance_agent, finance_answer_cache), name="Finance education"
        )
        if finance_agent is not None and finance_answer_cache is not None
        else finance_agent.graph.with_config(run_name="Finance education")
        if finance_agent is not None
        else RunnableLambda(empty_finance_qa_node),
        "finance",
    )
    add_agent_node(
        "portfolio_analysis_node",
        portfolio_agent.graph.with_config(run_name="Portfolio analysis")
        if portfolio_agent is not None
        else RunnableLambda(lambda state: {**portfolio_analysis_node(state), "status": "error"}),
        "portfolio",
    )
    add_agent_node(
        "market_analysis_node",
        build_clarification_graph(market_analysis_handler).with_config(run_name="Market analysis"),
        "market",
    )
    add_agent_node(
        "goals_node",
        goal_agent.graph.with_config(run_name="Goal planning")
        if goal_agent is not None
        else RunnableLambda(unavailable_goal_node),
        "goals",
    )
    add_agent_node(
        "news_synthesizer_node",
        news_agent.graph.with_config(run_name="News synthesis")
        if news_agent is not None
        else RunnableLambda(unavailable_news_node),
        "news",
    )
    add_agent_node(
        "tax_education_node",
        RunnableLambda(cached_handler(tax_agent, tax_answer_cache), name="Tax education")
        if tax_agent is not None and tax_answer_cache is not None
        else tax_agent.graph.with_config(run_name="Tax education")
        if tax_agent is not None
        else RunnableLambda(unavailable_tax_node),
        "tax",
    )
    workflow.add_node(
        "clarify_node", RunnableLambda(lambda state: {**clarify_node(state), "status": "answered"})
    )
    workflow.add_node(
        "unsupported_node",
        RunnableLambda(lambda state: {**unsupported_node(state), "status": "answered"}),
    )
    workflow.add_node("output_guardrail_node", RunnableLambda(output_guardrail_node))
    workflow.add_edge(START, "input_guardrail_node")
    workflow.add_conditional_edges(
        "input_guardrail_node",
        RunnableLambda(
            lambda state: (
                "output_guardrail_node" if state.get("guardrail_blocked") else "intent_router"
            ),
            name="Route after input guardrail",
        ),
    )
    workflow.add_conditional_edges(
        "intent_router",
        RunnableLambda(select_agent_route, name="Choose specialist agent"),
    )
    workflow.add_edge("finance_qa_node", "output_guardrail_node")
    workflow.add_edge("portfolio_analysis_node", "output_guardrail_node")
    workflow.add_edge("market_analysis_node", "output_guardrail_node")
    workflow.add_edge("goals_node", "output_guardrail_node")
    workflow.add_edge("news_synthesizer_node", "output_guardrail_node")
    workflow.add_edge("tax_education_node", "output_guardrail_node")
    workflow.add_edge("clarify_node", "output_guardrail_node")
    workflow.add_edge("unsupported_node", "output_guardrail_node")
    workflow.add_edge("output_guardrail_node", END)
    return workflow.compile(checkpointer=checkpointer)
