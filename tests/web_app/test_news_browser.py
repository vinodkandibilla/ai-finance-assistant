from pathlib import Path

from streamlit.testing.v1 import AppTest

from finassist.web_app.services.api_client import ApiError
from finassist.web_app.services.models import QueryResponse
from finassist.web_app.services.news import articles_from_response

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


def response():
    return QueryResponse(
        intent="news",
        answer="Supported report",
        citations=["https://example.com/a"],
        calculations=[
            {
                "type": "news_articles",
                "articles": [
                    {
                        "title": "Reported earnings",
                        "url": "https://example.com/a",
                        "publisher": "Example",
                        "published_at": None,
                        "summary": "A supported reported fact.",
                    },
                ],
            }
        ],
    )


def test_article_parser_deduplicates_and_rejects_unsafe_links():
    value = response()
    value.calculations[0]["articles"] *= 2
    value.calculations[0]["articles"].append(
        {"title": "Bad", "url": "javascript:bad()", "summary": "Bad"}
    )
    assert len(articles_from_response(value)) == 1


def test_news_search_retry_retains_results_and_ask_prefills(monkeypatch):
    calls = []

    def fetch(topic, period):
        calls.append((topic, period))
        if len(calls) == 2:
            raise ApiError("Offline", code="offline")
        return response()

    monkeypatch.setattr("finassist.web_app.components.news.fetch_news", fetch)
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.switch_page("views/news.py").run()

    def search():
        return next(button for button in app.button if button.label == "Search news")

    search().click().run()
    assert not app.exception
    assert calls == [("US Markets", "Past week")]
    assert any("Publication date unavailable" in item.value for item in app.caption)
    assert any(link.label == "Read article" for link in app.get("link_button"))
    search().click().run()
    assert not app.exception
    assert any("Showing previous results" in item.value for item in app.caption)
    next(b for b in app.button if b.label == "Retry news search").click().run()
    assert len(calls) == 3
    app.button(key="news_article_ask_0").click().run()
    assert "https://example.com/a" in app.chat_input[0].value
    assert not app.chat_message
