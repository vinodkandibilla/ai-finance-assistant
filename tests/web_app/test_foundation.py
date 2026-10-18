from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from finassist.web_app.config import WebSettings

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


@pytest.mark.parametrize(
    "page,title",
    [
        ("overview", "Overview"),
        ("finance_qa", "Finance Q&A"),
        ("portfolio", "Portfolio"),
        ("market", "Market"),
        ("goals", "Goals"),
        ("news", "News"),
        ("tax", "Tax Education"),
    ],
)
def test_navigation_without_backend(page: str, title: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Rendering must not contact the backend")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    app = AppTest.from_file(str(APP)).run()
    if page != "overview":
        app.switch_page(f"views/{page}.py").run()
    assert not app.exception
    assert any(element.value == title for element in app.title)
    assert any("For educational use only" in item.value for item in app.markdown)


def test_invalid_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINPILOT_API_BASE_URL", "file:///tmp/api")
    with pytest.raises(ValueError):
        WebSettings.from_environment()


def test_connection_error_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("Offline")

    monkeypatch.setattr(httpx.Client, "send", offline)
    app = AppTest.from_file(str(APP)).run()
    next(button for button in app.button if button.label == "Check API connection").click().run()
    assert not app.exception
    assert any("Backend unavailable" in element.value for element in app.error)
    app.switch_page("views/tax.py").run()
    assert any("Backend unavailable" in element.value for element in app.error)
    monkeypatch.setattr(
        httpx.Client,
        "send",
        lambda *a, **kw: httpx.Response(
            200,
            json={"status": "ok", "version": "0.1.0"},
            request=httpx.Request("GET", "http://api.test/health"),
        ),
    )
    next(button for button in app.button if button.label == "Check API connection").click().run()
    assert not app.error
    assert any("API connected" in element.value for element in app.success)
