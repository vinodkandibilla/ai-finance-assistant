"""Exercise actual landing-page actions and session-only question handoffs."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


@pytest.fixture
def landing(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Landing actions must not make backend calls")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx, "post", no_network)
    return AppTest.from_file(str(APP)).run()


@pytest.mark.parametrize(
    "label,title,question",
    [
        (
            "Explore Finance Education",
            "Finance Q&A",
            "Explain how ETFs work and the risks I should understand.",
        ),
        (
            "Explore Portfolio Analysis",
            "Portfolio",
            "Explain how to evaluate portfolio diversification and concentration.",
        ),
        ("Explore Market Intelligence", "Market", "What is the latest available price of AAPL?"),
        (
            "Explore Goal Planning",
            "Goals",
            "How much should I save monthly to reach $10,000 in two years?",
        ),
        (
            "Explore Financial News",
            "News",
            "Summarize the latest financial-market news with sources.",
        ),
        (
            "Explore Tax Education",
            "Tax Education",
            "Explain the difference between short-term and long-term capital gains.",
        ),
        ("Explore market context", "Market", "How should I interpret daily market index changes?"),
        (
            "Explore portfolio allocation",
            "Portfolio",
            "Explain a hypothetical 60% stocks, 30% bonds, 10% cash mix.",
        ),
    ],
)
def test_landing_actions(landing: AppTest, label: str, title: str, question: str) -> None:
    next(button for button in landing.button if button.label == label).click().run()
    assert not landing.exception
    assert landing.title[0].value == title
    assert landing.chat_input[0].value == question
    assert not landing.chat_message


def test_ask_ai_preserves_question_and_edits(landing: AppTest) -> None:
    landing.text_input[0].input("  How do bonds work?  ")
    next(button for button in landing.button if button.label == "Ask AI").click().run()
    assert not landing.exception
    assert landing.title[0].value == "Finance Q&A"
    assert landing.chat_input[0].value == "How do bonds work?"
    assert not landing.text_area
    assert not landing.chat_message


def test_empty_question_stays_on_overview(landing: AppTest) -> None:
    landing.text_input[0].input("   ")
    next(button for button in landing.button if button.label == "Ask AI").click().run()
    assert not landing.exception
    assert landing.title[0].value == "Overview"
    assert any("Enter a question" in item.value for item in landing.warning)


def test_previews_label_demo_figures(landing: AppTest) -> None:
    assert [metric.value for metric in landing.metric] == ["4,250.00", "$100,000"]
    text = " ".join(item.value for item in landing.caption)
    assert "Not live market data" in text
    assert "Synthetic account" in text
    assert "fictional" in text.lower()
