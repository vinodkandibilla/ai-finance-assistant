"""Pin router decisions independently of LLM predictions and evaluation scoring."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from finassist.core.config import Settings
from finassist.evaluation.router.evaluation import load_router_cases
from finassist.workflow import builder
from finassist.workflow.greetings import WELCOME

CASES = load_router_cases(Path(__file__).resolve().parents[1] / "evals/router/cases.jsonl")


@pytest.fixture(autouse=True)
def enable_market_quotes(monkeypatch):
    # Do not inherit routing behavior from the developer's .env or cached settings.
    settings = Settings(_env_file=None, market_quotes_enabled=True)
    monkeypatch.setattr(builder, "get_settings", lambda: settings)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
async def test_labeled_router_decision(case):
    classifier = AsyncMock()
    classifier.classify.return_value = case.expected_classified_intent

    result = await builder._classify_intent({"query": case.query}, classifier=classifier)

    assert result.get("classified_intent") == case.expected_classified_intent
    assert result.get("executed_handler") == case.expected_handler
    if case.expected_classified_intent is None:
        classifier.classify.assert_not_awaited()
        assert result["standalone_greeting"] is True
        assert result["answer"] == WELCOME
        assert result["status"] == "answered"
    else:
        classifier.classify.assert_awaited_once_with(case.query)
        assert result["intent"] == case.expected_classified_intent
        assert result["standalone_greeting"] is False


@pytest.mark.parametrize(
    "query",
    [
        "what is the stock price",
        "  WHAT IS THE STOCK PRICE?!  ",
        "What is the price of the stock?",
        "stock price",
        "price of stock!",
        "what is the price?",
        "market price.",
    ],
)
@pytest.mark.parametrize("enabled", [True, False], ids=["quotes-enabled", "quotes-disabled"])
def test_generic_market_decision_depends_on_feature_flag(monkeypatch, query, enabled):
    settings = Settings(_env_file=None, market_quotes_enabled=enabled)
    monkeypatch.setattr(builder, "get_settings", lambda: settings)
    expected = "clarify_node" if enabled else "market_analysis_node"
    assert builder._route_intent({"intent": "market", "query": query}) == expected


@pytest.mark.parametrize(
    "query",
    [
        "What is the stock price of AAPL?",
        "What is the price of the stock MSFT?",
        "Show market price for Apple.",
        "What is the price? AAPL",
    ],
)
def test_specific_market_query_is_not_redirected_by_generic_substring(query):
    assert builder._route_intent({"intent": "market", "query": query}) == "market_analysis_node"


@pytest.mark.parametrize(
    "intent,query,handler",
    [
        ("tax", "tax help", "tax_education_node"),
        ("portfolio", "Analyze my portfolio.", "portfolio_analysis_node"),
        ("goals", "How should I plan?", "goals_node"),
        ("news", "Give me news.", "news_synthesizer_node"),
        ("finance_qa", "Explain taxes and stock prices.", "finance_qa_node"),
        ("clarify", "What is AAPL's price?", "clarify_node"),
        ("other", "What is an ETF?", "unsupported_node"),
    ],
)
def test_routing_preserves_supplied_intent_despite_query_words(intent, query, handler):
    assert builder._route_intent({"intent": intent, "query": query}) == handler


@pytest.mark.parametrize("query", [None, "", "   "])
def test_missing_query_does_not_trigger_generic_market_redirect(query):
    assert builder._route_intent({"intent": "market", "query": query}) == "market_analysis_node"


@pytest.mark.parametrize(
    "query",
    [
        "Hi, what is an ETF?",
        "Hello, quote AAPL",
        "Good morning, show AAPL history",
    ],
)
async def test_greeting_with_task_reaches_classifier(query):
    classifier = AsyncMock()
    classifier.classify.return_value = "market"
    result = await builder._classify_intent({"query": query}, classifier=classifier)
    classifier.classify.assert_awaited_once_with(query)
    assert result["standalone_greeting"] is False
    assert result["executed_handler"] == "market_analysis_node"
