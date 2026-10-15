import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


def chat(monkeypatch: pytest.MonkeyPatch, responses: list) -> tuple[AppTest, list]:
    calls = []

    def send(client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, int):
            return httpx.Response(result, request=request, json={"detail": "private error"})
        return httpx.Response(200, request=request, json=result)

    monkeypatch.setattr(httpx.Client, "send", send)
    app = AppTest.from_file(str(APP)).run()
    app.switch_page("views/finance_qa.py").run()
    return app, calls


def button(app: AppTest, label: str) -> None:
    next(b for b in app.button if b.label == label).click().run()


@pytest.mark.parametrize(
    "intent,handler",
    [
        ("finance_qa", "finance_qa_node"),
        ("portfolio", "portfolio_analysis_node"),
        ("market", "market_analysis_node"),
        ("goals", "goals_node"),
        ("news", "news_synthesizer_node"),
        ("tax", "tax_education_node"),
    ],
)
def test_six_agents_and_safe_output(
    monkeypatch: pytest.MonkeyPatch, intent: str, handler: str
) -> None:
    app, calls = chat(
        monkeypatch,
        [
            {
                "intent": intent,
                "answer": "<script>bad()</script>",
                "classified_intent": intent,
                "executed_handler": handler,
                "citations": ["https://example.test/source", "javascript:bad()"],
                "calculation": {"total": 42},
                "calculations": [{"amount": 1}],
            }
        ],
    )
    app.chat_input[0].set_value("Explain this").run()
    assert not app.exception
    assert calls == [("/query", {"query": "Explain this"})]
    assert len(app.chat_message) == 2
    assert any(t.value == "<script>bad()</script>" for t in app.markdown)
    assert any("javascript:bad()" in t.value for t in app.text)
    assert any(handler in t.value for t in app.caption)
    assert len(app.json) == 2
    app.run()
    assert len(calls) == 1


def test_repeated_clarification_and_next_question(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(
        monkeypatch,
        [
            {
                "intent": "goals",
                "answer": "Need details",
                "clarification_question": "How much?",
                "thread_id": "t1",
            },
            {
                "intent": "goals",
                "answer": "Thanks",
                "clarification_question": "By when?",
                "thread_id": "t1",
            },
            {"intent": "goals", "answer": "Save monthly"},
            {"intent": "tax", "answer": "Tax answer"},
        ],
    )
    app.chat_input[0].set_value("Help me save").run()
    app.switch_page("views/goals.py").run()
    assert len(app.chat_message) == 2
    app.chat_input[0].set_value("10000").run()
    app.chat_input[0].set_value("Two years").run()
    app.chat_input[0].set_value("Explain taxes").run()
    assert not app.exception
    assert calls == [
        ("/query", {"query": "Help me save"}),
        ("/query/resume", {"thread_id": "t1", "reply": "10000"}),
        ("/query/resume", {"thread_id": "t1", "reply": "Two years"}),
        ("/query", {"query": "Explain taxes"}),
    ]


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError("private"), httpx.ReadTimeout("private"), 422]
)
def test_failure_recovery_without_duplicate_user(
    monkeypatch: pytest.MonkeyPatch, failure: object
) -> None:
    app, calls = chat(monkeypatch, [failure, {"intent": "finance_qa", "answer": "Recovered"}])
    app.chat_input[0].set_value("What is an ETF?").run()
    assert app.error
    assert app.chat_input[0].disabled
    assert len(app.chat_message) == 1
    assert "private" not in app.error[0].value
    button(app, "Retry last request")
    assert not app.exception
    assert not app.error
    assert len(app.chat_message) == 2
    assert calls[0] == calls[1]


def test_resume_failure_retains_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(
        monkeypatch,
        [
            {
                "intent": "market",
                "answer": "",
                "clarification_question": "Which symbol?",
                "thread_id": "t2",
            },
            503,
            {"intent": "market", "answer": "Quote"},
        ],
    )
    app.chat_input[0].set_value("Get a quote").run()
    app.chat_input[0].set_value("AAPL").run()
    button(app, "Retry last request")
    assert not app.exception
    assert calls[1:] == [("/query/resume", {"thread_id": "t2", "reply": "AAPL"})] * 2
    assert len(app.chat_message) == 4


def test_new_chat_clears_thread_messages_and_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, calls = chat(
        monkeypatch,
        [
            {"intent": "tax", "answer": "", "clarification_question": "Year?", "thread_id": "t3"},
            403,
            {"intent": "news", "answer": "News"},
        ],
    )
    app.chat_input[0].set_value("Tax question").run()
    app.chat_input[0].set_value("2026").run()
    app.button(key="new_chat").click().run()
    app.button(key="confirm_new_chat").click().run()
    assert not app.exception
    assert not app.chat_message
    assert not app.error
    assert not app.chat_input[0].disabled
    assert not app.chat_input[0].value
    app.chat_input[0].set_value("News question").run()
    assert calls[-1] == ("/query", {"query": "News question"})


def test_single_composer_sends_question(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(monkeypatch, [{"intent": "finance_qa", "answer": "Bonds"}])
    assert not app.text_area
    assert len(app.chat_input) == 1
    app.chat_input[0].set_value("Explain bonds").run()
    assert not app.exception
    assert calls == [("/query", {"query": "Explain bonds"})]
    assert not app.chat_input[0].value


def test_missing_clarification_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(
        monkeypatch, [{"intent": "goals", "answer": "", "clarification_question": "When?"}]
    )
    app.chat_input[0].set_value("Help plan").run()
    assert not app.exception
    assert "without a thread" in app.error[0].value
    assert len(calls) == 1


def test_application_error_is_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = chat(
        monkeypatch, [{"intent": "market", "answer": "No quote available", "status": "error"}]
    )
    app.chat_input[0].set_value("AAPL price").run()
    assert not app.exception
    assert any("could not complete" in w.value for w in app.warning)
    assert not app.chat_input[0].disabled


def test_edit_failed_message_preserves_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(monkeypatch, [422, {"intent": "finance_qa", "answer": "Answer"}])
    app.chat_input[0].set_value("Original question").run()
    button(app, "Edit failed message")
    assert not app.exception
    assert not app.error
    assert app.chat_input[0].value == "Original question"
    app.chat_input[0].set_value("Revised question").run()
    assert not app.exception
    assert calls[-1] == ("/query", {"query": "Revised question"})
    assert len(app.chat_message) == 3


def test_markdown_citations_render_with_html_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Market explanation. [E2](<https://news.test/story>)\n\n**Reported context**"
    app, _ = chat(
        monkeypatch,
        [{"intent": "news", "answer": answer, "citations": ["https://news.test/story"]}],
    )
    app.chat_input[0].set_value("Explain market news").run()
    rendered = next(
        item for item in app.markdown if item.value == answer.replace("[E2]", "[news.test]")
    )
    assert rendered.proto.allow_html is False
    assert app.get("link_button")[0].label == "https://news.test/story"
    assert len(app.get("link_button")) == 1
    assert not app.exception


def test_new_chat_cancel_preserves_messages_and_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(monkeypatch, [{"intent": "finance_qa", "answer": "Bonds"}])
    app.chat_input[0].set_value("Explain bonds").run()
    app.session_state["question_finance_qa"] = "My next question"
    app.run()
    app.button(key="new_chat").click().run()
    assert len(app.chat_message) == 2
    app.button(key="cancel_new_chat").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert not app.text_area
    assert len(calls) == 1


def test_empty_new_chat_skips_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    app, calls = chat(monkeypatch, [])
    app.button(key="new_chat").click().run()
    assert not app.exception
    assert not any(b.key == "confirm_new_chat" for b in app.button)
    assert not calls


def test_question_renders_before_http_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from finassist.web_app.components import chat as chat_ui

    app, calls = chat(monkeypatch, [{"intent": "finance_qa", "answer": "Bonds explained"}])
    events: list[str] = []
    original_render = chat_ui._render_message
    original_send = httpx.Client.send

    def render(message: chat_ui.Message) -> None:
        if message.role == "user":
            events.append("question rendered")
        original_render(message)

    def send(client: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        events.append("HTTP request")
        assert "question rendered" in events[:-1]
        return original_send(client, request, **kwargs)

    monkeypatch.setattr(chat_ui, "_render_message", render)
    monkeypatch.setattr(httpx.Client, "send", send)
    app.chat_input[0].set_value("Explain bonds").run()
    assert not app.exception
    assert len(calls) == 1
    assert len(app.chat_message) == 2
    assert app.session_state["conversation"].pending_request is None
