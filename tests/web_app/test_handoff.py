"""Final architectural and session invariants for the capstone handoff."""

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from finassist.web_app.components.chat import ChatState
from finassist.web_app.services.models import ResumeRequest
from finassist.web_app.services.showcase import project_goal

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src/finassist/web_app/app.py"


def test_frontend_imports_no_backend_implementation() -> None:
    for path in (ROOT / "src/finassist/web_app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            assert all(
                not module.startswith("finassist.") or module.startswith("finassist.web_app")
                for module in modules
            ), path


@pytest.mark.parametrize("rate,months", [(0, 1), (0, 120), (6, 36), (20, 600)])
def test_required_contribution_funds_scenario(rate: float, months: int) -> None:
    required = project_goal(100000, 1000, 0, rate, months).required_monthly
    assert project_goal(100000, 1000, required, rate, months).balances[-1] >= 100000


def test_watchlist_isolated_between_sessions() -> None:
    first = AppTest.from_file(str(APP), default_timeout=10)
    first.session_state["market_watchlist"] = ["AAPL"]
    first.run().switch_page("views/market.py").run()
    second = AppTest.from_file(str(APP), default_timeout=10).run()
    second.switch_page("views/market.py").run()
    assert not second.exception
    assert not any(s.label.startswith("Saved symbols") for s in second.selectbox)


def test_pending_clarification_disables_new_showcase_requests() -> None:
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["conversation"] = ChatState(thread_id="pending-thread")
    app.run().switch_page("views/news.py").run()
    assert next(b for b in app.button if b.label == "Get news summary").disabled
    app.switch_page("views/market.py").run()
    assert app.text_input(key="market_chart_symbol").disabled
    app.switch_page("views/goals.py").run()
    assert next(b for b in app.button if b.label == "Ask AI about this scenario").disabled


def test_new_chat_removes_queued_action_and_request_scope() -> None:
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["conversation"] = ChatState(
        thread_id="pending",
        failed_request=ResumeRequest(thread_id="pending", reply="old"),
        error="Offline",
    )
    app.session_state["queued_question"] = "Do not send this"
    app.session_state["portfolio_scope_customer"] = "C001"
    app.run().switch_page("views/finance_qa.py").run()
    app.button(key="new_chat").click().run()
    app.button(key="confirm_new_chat").click().run()
    assert not app.exception
    assert app.session_state["conversation"].thread_id is None
    assert app.session_state["conversation"].failed_request is None
    assert "queued_question" not in app.session_state
    assert "portfolio_scope_customer" not in app.session_state


def test_theme_text_and_action_contrast() -> None:
    import tomllib

    theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["theme"]

    def luminance(color: str) -> float:
        values = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return sum(v * w for v, w in zip(linear, [0.2126, 0.7152, 0.0722], strict=True))

    for foreground, background in [
        (theme["textColor"], theme["backgroundColor"]),
        (theme["sidebar"]["textColor"], theme["sidebar"]["backgroundColor"]),
        (theme["linkColor"], theme["backgroundColor"]),
        ("#FFFFFF", theme["primaryColor"]),
    ]:
        low, high = sorted([luminance(foreground), luminance(background)])
        assert (high + 0.05) / (low + 0.05) >= 4.5


@pytest.mark.parametrize(
    "page", ["overview", "finance_qa", "portfolio", "market", "goals", "news", "tax"]
)
def test_workspace_shell_and_adjustable_chat(page: str) -> None:
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    if page != "overview":
        app.switch_page(f"views/{page}.py").run()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert len(app.title) == 1
    assert not any(item.key == "chat_panel_width" for item in app.select_slider)
    app.session_state[f"question_{'finance_qa' if page == 'overview' else page}"] = (
        "Keep this draft while navigating"
    )
    app.run()
    assert not app.exception
    assert app.chat_input[0].value == "Keep this draft while navigating"
    assert not app.text_area
