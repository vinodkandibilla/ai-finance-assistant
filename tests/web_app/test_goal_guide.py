"""Goal examples hand off drafts without sending requests or changing calculator inputs."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src/finassist/web_app/app.py"


@pytest.mark.parametrize("question_index", range(4))
def test_goal_question_prefills_chat_without_provider_call(monkeypatch, question_index):
    def no_network(*args, **kwargs):
        pytest.fail("Choosing a suggested goal question must not call the API")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.switch_page("views/goals.py").run()
    app.selectbox(key="goal_example_topic").set_value("Home down payment").run()
    app.button(key=f"goal_question_{question_index}").click().run()
    assert not app.exception
    assert "home down payment" in app.chat_input[0].value
    assert "10,000" in app.chat_input[0].value
    assert not app.chat_message
    assert app.number_input[0].value == 10000.0
    assert len(app.get("plotly_chart")) == 1
