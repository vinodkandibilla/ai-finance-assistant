from ai_finance_assistant.graph.builder import (
    CLARIFY_RESPONSE,
    FINANCE_QA_RESPONSE,
    GOALS_RESPONSE,
    MARKET_ANALYSIS_RESPONSE,
    NEWS_RESPONSE,
    PORTFOLIO_ANALYSIS_RESPONSE,
    TAX_RESPONSE,
    build_graph,
)
from ai_finance_assistant.graph.state import FinanceIntent
from ai_finance_assistant.rag.models import RetrievedDocument


class FakeClassifier:
    def __init__(self, intent: FinanceIntent) -> None:
        self.intent = intent

    async def classify(self, query: str) -> FinanceIntent:
        return self.intent


class FakeRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents

    async def search(self, query: str, *, limit: int = 5) -> list[RetrievedDocument]:
        return self.documents


class FakeFinanceLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def ainvoke(self, prompt: object) -> str:
        return self.answer


async def test_graph_routes_finance_qa_intent_to_qa_node() -> None:
    result = await build_graph(FakeClassifier("finance_qa")).ainvoke({"query": "What is an ETF?"})

    assert result["intent"] == "finance_qa"
    assert result["answer"] == FINANCE_QA_RESPONSE


async def test_graph_uses_retrieved_context_for_finance_qa() -> None:
    retriever = FakeRetriever(
        [
            RetrievedDocument(
                content="An ETF is a basket of securities that trades on an exchange.",
                source="handbook.pdf",
                metadata={"chapter": "4", "section": "2", "page_number": "25"},
                score=0.92,
            )
        ]
    )
    llm = FakeFinanceLLM("ETF means an exchange-traded fund that holds a basket of assets.")

    result = await build_graph(FakeClassifier("finance_qa"), retriever=retriever, llm=llm).ainvoke(
        {"query": "What is an ETF?"}
    )

    assert result["intent"] == "finance_qa"
    assert result["answer"] == "ETF means an exchange-traded fund that holds a basket of assets."


async def test_graph_includes_citation_metadata_in_finance_answer() -> None:
    retriever = FakeRetriever(
        [
            RetrievedDocument(
                content="An ETF is a basket of securities that trades on an exchange.",
                source="handbook.pdf",
                metadata={"chapter": "4", "section": "2", "page_number": "25"},
                score=0.92,
            )
        ]
    )
    llm = FakeFinanceLLM("ETF is a basket of securities. Citation: Chapter 4, Section 2, Page 25.")

    result = await build_graph(FakeClassifier("finance_qa"), retriever=retriever, llm=llm).ainvoke(
        {"query": "What is an ETF?"}
    )

    assert result["intent"] == "finance_qa"
    assert "Chapter 4" in result["answer"]
    assert "Section 2" in result["answer"]
    assert "Page 25" in result["answer"]


async def test_graph_routes_other_intent_to_unsupported_node() -> None:
    result = await build_graph(FakeClassifier("other")).ainvoke({"query": "What is an ETF?"})

    assert result["intent"] == "other"
    assert result["answer"] == "This request is not supported yet."


async def test_graph_routes_portfolio_intent_to_computational_node() -> None:
    result = await build_graph(FakeClassifier("portfolio")).ainvoke(
        {"query": "Analyze my portfolio allocation and concentration."}
    )

    assert result["intent"] == "portfolio"
    assert result["answer"] == PORTFOLIO_ANALYSIS_RESPONSE


async def test_graph_routes_market_intent_to_market_analysis_node() -> None:
    result = await build_graph(FakeClassifier("market")).ainvoke(
        {"query": "What is the current price of AAPL?"}
    )

    assert result["intent"] == "market"
    assert result["answer"] == "Your stock AAPL is $333.0"
    assert result["answer"] == MARKET_ANALYSIS_RESPONSE


async def test_graph_routes_goals_intent_to_computational_node() -> None:
    result = await build_graph(FakeClassifier("goals")).ainvoke(
        {"query": "How much should I save each month?"}
    )

    assert result["intent"] == "goals"
    assert result["answer"] == GOALS_RESPONSE


async def test_graph_routes_news_intent_to_news_synthesizer_node() -> None:
    result = await build_graph(FakeClassifier("news")).ainvoke(
        {"query": "Give me a concise summary of today's top AI news."}
    )

    assert result["intent"] == "news"
    assert result["answer"] == NEWS_RESPONSE


async def test_graph_routes_tax_intent_to_tax_education_node() -> None:
    result = await build_graph(FakeClassifier("tax")).ainvoke(
        {"query": "What is the standard deduction for Married Filing Jointly with two qualifying children for the year 2026?"}
    )

    assert result["intent"] == "tax"
    assert result["answer"] == TAX_RESPONSE


async def test_graph_routes_market_query_without_ticker_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("market")).ainvoke(
        {"query": "What is the stock price?"}
    )

    assert result["intent"] == "market"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_market_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("market")).ainvoke(
        {"query": "What is the stock price?"}
    )

    assert result["intent"] == "market"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_news_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("news")).ainvoke(
        {"query": "top news"}
    )

    assert result["intent"] == "news"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_tax_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("tax")).ainvoke(
        {"query": "tax details"}
    )

    assert result["intent"] == "tax"
    assert result["answer"] == CLARIFY_RESPONSE


async def test_graph_routes_generic_or_incomplete_query_to_clarify_node() -> None:
    result = await build_graph(FakeClassifier("clarify")).ainvoke(
        {"query": "What is the stock price?"}
    )

    assert result["intent"] == "clarify"
    assert result["answer"] == CLARIFY_RESPONSE
