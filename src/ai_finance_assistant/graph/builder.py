from typing import Literal

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_finance_assistant.agents.common.clarification import build_clarification_graph
from ai_finance_assistant.agents.finance_qa import FinanceQaAgent
from ai_finance_assistant.agents.goal_planning import GoalPlanningAgent
from ai_finance_assistant.agents.market_analysis.agent import MarketAnalysisAgent
from ai_finance_assistant.agents.news_synthesizer import NewsSynthesizerAgent
from ai_finance_assistant.agents.portfolio_analysis.agent import PortfolioAnalysisAgent
from ai_finance_assistant.agents.tax_education.graph import TaxEducationAgent
from ai_finance_assistant.core.config import get_settings
from ai_finance_assistant.graph.classifier import IntentClassifier
from ai_finance_assistant.graph.state import FinanceAssistantState

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
    intent = await classifier.classify(state["query"])
    handler = _route_intent({**state, "intent": intent})
    return {"intent": intent, "classified_intent": intent, "executed_handler": handler}


def _is_generic_query(intent: str | None, query: str | None) -> bool:
    if not query:
        return False

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
    tax_agent: TaxEducationAgent | None = None,
    goal_agent: GoalPlanningAgent | None = None,
    news_agent: NewsSynthesizerAgent | None = None,
    portfolio_agent: PortfolioAnalysisAgent | None = None,
    checkpointer: object | None = None,
) -> FinanceAssistantGraph:
    async def classify_node(state: FinanceAssistantState) -> FinanceAssistantState:
        return await _classify_intent(state, classifier=classifier)

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

    workflow: StateGraph[
        FinanceAssistantState, None, FinanceAssistantState, FinanceAssistantState
    ] = StateGraph(FinanceAssistantState)
    workflow.add_node("intent_router", classify_node)
    workflow.add_node(
        "finance_qa_node",
        finance_agent.graph if finance_agent is not None else RunnableLambda(empty_finance_qa_node),
    )
    workflow.add_node(
        "portfolio_analysis_node",
        portfolio_agent.graph
        if portfolio_agent is not None
        else RunnableLambda(lambda state: {**portfolio_analysis_node(state), "status": "error"}),
    )
    workflow.add_node("market_analysis_node", build_clarification_graph(market_analysis_handler))
    workflow.add_node(
        "goals_node",
        goal_agent.graph if goal_agent is not None else RunnableLambda(unavailable_goal_node),
    )
    workflow.add_node(
        "news_synthesizer_node",
        news_agent.graph if news_agent is not None else RunnableLambda(unavailable_news_node),
    )
    workflow.add_node(
        "tax_education_node",
        tax_agent.graph if tax_agent is not None else RunnableLambda(unavailable_tax_node),
    )
    workflow.add_node(
        "clarify_node", RunnableLambda(lambda state: {**clarify_node(state), "status": "answered"})
    )
    workflow.add_node(
        "unsupported_node",
        RunnableLambda(lambda state: {**unsupported_node(state), "status": "answered"}),
    )
    workflow.add_edge(START, "intent_router")
    workflow.add_conditional_edges(
        "intent_router",
        lambda state: state["executed_handler"],
    )
    workflow.add_edge("finance_qa_node", END)
    workflow.add_edge("portfolio_analysis_node", END)
    workflow.add_edge("market_analysis_node", END)
    workflow.add_edge("goals_node", END)
    workflow.add_edge("news_synthesizer_node", END)
    workflow.add_edge("tax_education_node", END)
    workflow.add_edge("clarify_node", END)
    workflow.add_edge("unsupported_node", END)
    return workflow.compile(checkpointer=checkpointer)
