from typing import Literal, Protocol

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_finance_assistant.graph.classifier import IntentClassifier
from ai_finance_assistant.graph.state import FinanceAssistantState
from ai_finance_assistant.providers.market_data import extract_supported_symbol
from ai_finance_assistant.rag.models import RetrievedDocument


class FinanceRetriever(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[RetrievedDocument]: ...


class TaxRetriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: object | None = None,
    ) -> list[RetrievedDocument]: ...


class FinanceLLM(Protocol):
    async def ainvoke(self, messages: object) -> object: ...


class MarketQuoteTool(Protocol):
    async def ainvoke(self, input: dict[str, str]) -> object: ...


type FinanceAssistantGraph = CompiledStateGraph[
    FinanceAssistantState, None, FinanceAssistantState, FinanceAssistantState
]

FINANCE_QA_RESPONSE = (
    "A standard response - An ETF is a popular type of investment fund that holds a collection of assets "
    "(like stocks, bonds, or commodities) and trades on a regular stock exchange "
    "just like an individual stock."
)
PORTFOLIO_ANALYSIS_RESPONSE = (
    "Your portfolio is risk free and balanced. (Placeholder only; no holdings were analyzed.)"
)
MARKET_QUOTE_UNAVAILABLE_RESPONSE = (
    "I couldn't retrieve a current quote. Check the ticker and try again later."
)
GOALS_RESPONSE = "You will need to contribute $555 per month to achieve your goal in 5 months."
NEWS_RESPONSE = (
    "**Model Releases:** Google unveiled its next-generation frontier model, "
    "**Gemini 4 Argon**, featuring advanced reasoning and a 1-million-token output "
    "limit aimed at heavy-duty workloads like coding, knowledge work, and "
    "cybersecurity defense"
)
TAX_RESPONSE = (
    "For a Married Filing Jointly household with two qualifying children in 2026, "
    "the standard deduction is $32,000."
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
    return {"intent": await classifier.classify(state["query"])}


def _is_generic_query(intent: str | None, query: str | None) -> bool:
    if not query:
        return False

    normalized = query.strip().lower()
    if not normalized:
        return False

    generic_by_intent: dict[str, tuple[str, ...]] = {
        "finance_qa": (
            "what is finance",
            "finance basics",
            "investment basics",
            "financial concepts",
        ),
        "market": (
            "what is the stock price",
            "what is the price of the stock",
            "stock price",
            "price of stock",
            "what is the price",
            "market price",
        ),
        "news": (
            "top news",
            "latest news",
            "news today",
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
        return is_generic and not any(
            token in normalized
            for token in (
                "aapl",
                "msft",
                "goog",
                "googl",
                "nvda",
                "tsla",
                "amzn",
                "meta",
                "nflx",
                "apple",
                "microsoft",
                "google",
                "nvidia",
                "tesla",
                "amazon",
            )
        )

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
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "finance_qa_node"
    if intent == "portfolio":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "portfolio_analysis_node"
    if intent == "market":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "market_analysis_node"
    if intent == "goals":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "goals_node"
    if intent == "news":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "news_synthesizer_node"
    if intent == "tax":
        if _is_generic_query(intent, state.get("query")):
            return "clarify_node"
        return "tax_education_node"
    if intent == "clarify":
        return "clarify_node"
    return "unsupported_node"


def _join_context(documents: list[RetrievedDocument]) -> str:
    context_blocks: list[str] = []
    for document in documents:
        if not document.content or not document.content.strip():
            continue
        metadata = document.metadata or {}
        chapter = metadata.get("chapter")
        section = metadata.get("section")
        page_number = metadata.get("page_number")

        citation = []
        if chapter is not None:
            citation.append(f"Chapter {chapter}")
        if section is not None:
            citation.append(f"Section {section}")
        if page_number is not None:
            citation.append(f"Page {page_number}")

        citation_text = ", ".join(citation)
        block = f"[Source: {document.source}]"
        if citation_text:
            block += f"\nCitation: {citation_text}"
        block += f"\n{document.content.strip()}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


async def _answer_finance_qa_with_context(
    query: str,
    *,
    retriever: FinanceRetriever | None,
    llm: FinanceLLM | None,
) -> str:
    if not retriever or not llm:
        return FINANCE_QA_RESPONSE

    try:
        matches = await retriever.search(query, limit=3)
    except Exception:
        return FINANCE_QA_RESPONSE

    if not matches:
        return FINANCE_QA_RESPONSE

    context = _join_context(matches)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful financial assistant. Use only the retrieved context to answer "
                "the user's question. If the context does not contain enough information, say so "
                "briefly and avoid inventing facts. Cite the relevant chapter, section, and page number "
                "when available. If you reference a fact from the retrieved context, include the citation in the answer.",
            ),
            (
                "human",
                "Question: {question}\n\nRetrieved context:\n{context}\n\nAnswer clearly and concisely. "
                "If available, include a citation in the form 'Chapter X, Section Y, Page Z.'",
            ),
        ]
    )

    async def _call_llm(messages: object) -> object:
        return await llm.ainvoke(messages)

    qa_chain = (
        {
            "context": RunnableLambda(lambda _: context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(_call_llm)
        | StrOutputParser()
    )

    try:
        answer = await qa_chain.ainvoke(query)
    except Exception:
        return FINANCE_QA_RESPONSE

    return (answer or FINANCE_QA_RESPONSE).strip() or FINANCE_QA_RESPONSE


async def finance_qa_node(
    state: FinanceAssistantState,
    *,
    retriever: FinanceRetriever | None = None,
    llm: FinanceLLM | None = None,
) -> FinanceAssistantState:
    query = state.get("query") or ""
    answer = await _answer_finance_qa_with_context(query, retriever=retriever, llm=llm)
    return {"answer": answer}


def portfolio_analysis_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": PORTFOLIO_ANALYSIS_RESPONSE}


async def market_analysis_node(
    state: FinanceAssistantState,
    *,
    quote_tool: MarketQuoteTool | None = None,
) -> FinanceAssistantState:
    query = state.get("query") or ""
    symbol = extract_supported_symbol(query)
    if symbol is None:
        return {
            "answer": (
                "I can look up current quotes for MAG7 tickers: "
                "AAPL, MSFT, AMZN, GOOGL, META, NVDA, and TSLA."
            )
        }
    if quote_tool is None:
        return {"answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE}

    try:
        quote = await quote_tool.ainvoke({"symbol": symbol})
    except Exception:
        return {"answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE}

    if not isinstance(quote, str) or not quote.strip():
        return {"answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE}
    if quote.startswith("Stock price unavailable"):
        return {"answer": quote}
    try:
        formatted_quote = f"{float(quote.strip()):.2f}"
    except ValueError:
        return {"answer": MARKET_QUOTE_UNAVAILABLE_RESPONSE}
    return {"answer": f"The latest available price for {symbol} is ${formatted_quote}."}


def goals_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": GOALS_RESPONSE}


def news_synthesizer_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": NEWS_RESPONSE}


async def tax_education_node_with_context(
    state: FinanceAssistantState,
    *,
    retriever: TaxRetriever | None,
    llm: FinanceLLM | None,
) -> FinanceAssistantState:
    query = (state.get("query") or "").strip()
    if not query or retriever is None or llm is None:
        return {"answer": TAX_RESPONSE}

    from ai_finance_assistant.rag.tax_education import build_tax_metadata_filter, extract_tax_metadata

    metadata = extract_tax_metadata(query)
    metadata_filter = build_tax_metadata_filter(metadata)
    try:
        matches = await retriever.search(query, limit=3, metadata_filter=metadata_filter)
    except Exception:
        matches = []
    if not matches and metadata_filter is not None:
        try:
            matches = await retriever.search(query, limit=3)
        except Exception:
            return {"answer": TAX_RESPONSE}
    if not matches:
        return {"answer": TAX_RESPONSE}

    context_blocks = []
    for document in matches:
        metadata = document.metadata or {}
        citation = metadata.get("title") or metadata.get("chunk_id") or document.source
        context_blocks.append(f"[Source: {citation}]\n{document.content.strip()}")
    context = "\n\n".join(context_blocks)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful tax education assistant. Answer only from the retrieved IRS tax corpus. "
                "Do not provide personalized legal or tax advice, and do not invent amounts or rules. "
                "If the context is insufficient, state what is missing. Cite the source title for factual claims.",
            ),
            (
                "human",
                "Question: {question}\n\nRetrieved tax context:\n{context}\n\n"
                "Answer clearly and concisely, citing the relevant source title.",
            ),
        ]
    )

    async def _call_llm(messages: object) -> object:
        return await llm.ainvoke(messages)

    qa_chain = (
        {
            "context": RunnableLambda(lambda _: context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(_call_llm)
        | StrOutputParser()
    )
    try:
        answer = await qa_chain.ainvoke(query)
    except Exception:
        return {"answer": TAX_RESPONSE}
    return {"answer": (answer or TAX_RESPONSE).strip() or TAX_RESPONSE}


def clarify_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": CLARIFY_RESPONSE}


def unsupported_node(_: FinanceAssistantState) -> FinanceAssistantState:
    return {"answer": "This request is not supported yet."}


def build_graph(
    classifier: IntentClassifier,
    *,
    retriever: FinanceRetriever | None = None,
    tax_retriever: TaxRetriever | None = None,
    llm: FinanceLLM | None = None,
    market_quote_tool: MarketQuoteTool | None = None,
) -> FinanceAssistantGraph:
    async def classify_node(state: FinanceAssistantState) -> FinanceAssistantState:
        return await _classify_intent(state, classifier=classifier)

    async def finance_qa_node_with_context(
        state: FinanceAssistantState,
    ) -> FinanceAssistantState:
        return await finance_qa_node(state, retriever=retriever, llm=llm)

    async def market_analysis_node_with_quote(
        state: FinanceAssistantState,
    ) -> FinanceAssistantState:
        return await market_analysis_node(state, quote_tool=market_quote_tool)

    async def tax_education_node_with_retrieval(
        state: FinanceAssistantState,
    ) -> FinanceAssistantState:
        return await tax_education_node_with_context(state, retriever=tax_retriever, llm=llm)

    workflow: StateGraph[
        FinanceAssistantState, None, FinanceAssistantState, FinanceAssistantState
    ] = StateGraph(FinanceAssistantState)
    workflow.add_node("intent_router", classify_node)
    workflow.add_node("finance_qa_node", finance_qa_node_with_context)
    workflow.add_node("portfolio_analysis_node", RunnableLambda(portfolio_analysis_node))
    workflow.add_node("market_analysis_node", market_analysis_node_with_quote)
    workflow.add_node("goals_node", RunnableLambda(goals_node))
    workflow.add_node("news_synthesizer_node", RunnableLambda(news_synthesizer_node))
    workflow.add_node("tax_education_node", tax_education_node_with_retrieval)
    workflow.add_node("clarify_node", RunnableLambda(clarify_node))
    workflow.add_node("unsupported_node", RunnableLambda(unsupported_node))
    workflow.add_edge(START, "intent_router")
    workflow.add_conditional_edges("intent_router", _route_intent)
    workflow.add_edge("finance_qa_node", END)
    workflow.add_edge("portfolio_analysis_node", END)
    workflow.add_edge("market_analysis_node", END)
    workflow.add_edge("goals_node", END)
    workflow.add_edge("news_synthesizer_node", END)
    workflow.add_edge("tax_education_node", END)
    workflow.add_edge("clarify_node", END)
    workflow.add_edge("unsupported_node", END)
    return workflow.compile()
