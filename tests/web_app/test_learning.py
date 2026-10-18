"""Guided topic navigation and inline learning flows through the actual HTTP adapter."""

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from finassist.web_app.components.learning import TOPICS

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


@pytest.mark.parametrize("page", ["finance_qa", "tax"])
def test_every_topic_and_suggested_question(page: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise AssertionError("Topic and chip selection must not call the API")

    monkeypatch.setattr(httpx.Client, "send", offline)
    app = AppTest.from_file(str(APP)).run()
    app.switch_page(f"views/{page}.py").run()
    assert app.selectbox[0].value == ("Tax Education" if page == "tax" else "Investing Basics")
    for topic in TOPICS:
        app.selectbox[0].select(topic.name).run()
        assert not app.exception
        assert any(item.value == topic.introduction for item in app.markdown)
        for question in topic.questions:
            next(b for b in app.button if b.label == question).click().run()
            assert app.chat_input[0].value == question
            assert "focus_chat_draft" not in app.session_state
            assert any("focusEditor" in element.proto.body for element in app.get("html"))
            assert not app.chat_message


@pytest.mark.parametrize(
    "citations", [None, [], ["Handbook chapter 1", "https://example.test/source"]]
)
def test_inline_answer_sources_and_clarification(
    citations: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    results = [
        {
            "intent": "tax",
            "classified_intent": "tax",
            "executed_handler": "tax_education_node",
            "answer": "Please specify a year.",
            "citations": citations,
            "clarification_question": "Which tax year?",
            "thread_id": "learning-thread",
        },
        {"intent": "tax", "answer": "Here is an educational explanation.", "citations": citations},
        {"intent": "finance_qa", "answer": "A comparison."},
    ]

    def send(client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, request=request, json=results.pop(0))

    monkeypatch.setattr(httpx.Client, "send", send)
    app = AppTest.from_file(str(APP)).run()
    app.switch_page("views/finance_qa.py").run()
    question = TOPICS[0].questions[0]
    next(b for b in app.button if b.label == question).click().run()
    app.chat_input[0].set_value(question).run()
    assert not app.exception
    assert calls == [("/query", {"query": question})]
    assert any("tax_education_node" in item.value for item in app.caption)
    assert not any("Agent: finance_qa_node" in item.value for item in app.caption)
    if citations:
        assert any(item.label == "Sources" for item in app.expander)
        assert any("Handbook chapter 1" in item.value for item in app.text)
    else:
        assert not any(item.label == "Sources" for item in app.expander)
        assert any("No sources were supplied" in item.value for item in app.caption)
    assert all(b.disabled for b in app.button if b.key and b.key.startswith("learn_"))
    app.selectbox[0].select("Tax Education").run()
    assert len(app.chat_message) == 2
    app.chat_input[0].set_value("2026").run()
    assert calls[-1] == ("/query/resume", {"thread_id": "learning-thread", "reply": "2026"})
    assert len(app.chat_message) == 4
    app.chat_input[0].set_value("Compare deductions and credits for US tax year 2026.").run()
    assert calls[-1][0] == "/query"
    assert len(app.chat_message) == 6
    assert not app.exception
