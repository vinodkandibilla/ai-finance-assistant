"""Run with streamlit run src/finassist/web_app/app.py from the repository root."""

import streamlit as st

from finassist.web_app.components.chat import conversational_screen
from finassist.web_app.components.layout import (
    compact_spacing,
    educational_disclaimer,
    error_state,
    loading_state,
    page_header,
    status_indicator,
)
from finassist.web_app.components.split_pane import resizable_chat_divider
from finassist.web_app.config import WebSettings
from finassist.web_app.services.api_client import ApiError, check_health

st.set_page_config(page_title="FinPilot AI", page_icon="✈️", layout="wide")
compact_spacing()
with st.sidebar:
    st.subheader("✈ FinPilot AI")
    st.caption("Learn. Explore. Plan.")
    if st.button("Check API connection", width="stretch"):
        with loading_state("Checking connection…"):
            try:
                health = check_health(WebSettings.from_environment())
            except (ApiError, ValueError) as error:
                st.session_state["api_connection"] = ("error", str(error))
            else:
                st.session_state["api_connection"] = (
                    "success",
                    f"API connected · v{health.version}",
                )
    connection = st.session_state.get("api_connection")
    if connection:
        state, message = connection
        if state == "error":
            error_state(message)
        else:
            status_indicator(message, "success")
    else:
        st.caption("API connection has not been checked.")
    st.caption("Educational workspace · Session only")

page = st.navigation(
    [
        st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
        st.Page("views/finance_qa.py", title="Finance Q&A", icon="💬"),
        st.Page("views/portfolio.py", title="Portfolio", icon="💼"),
        st.Page("views/market.py", title="Market", icon="📈"),
        st.Page("views/goals.py", title="Goals", icon="🎯"),
        st.Page("views/news.py", title="News", icon="📰"),
        st.Page("views/tax.py", title="Tax Education", icon="📚"),
    ]
)
workspace = page.url_path or "overview"
# The default page's URL is empty; every other page uses its script stem.
if workspace not in {"finance_qa", "portfolio", "market", "goals", "news", "tax"}:
    workspace = "overview"
descriptions = {
    "overview": "Explore financial insights and ask FinPilot alongside them.",
    "finance_qa": "Build your understanding of investing and personal finance.",
    "portfolio": "Explore portfolio allocation, concentration, and risk.",
    "market": "Explore market questions and available quote context.",
    "goals": "Plan savings goals and explore financial projections.",
    "news": "Explore financial news and cited summaries.",
    "tax": "Learn about US taxes with educational guidance.",
}
with st.container(key="finpilot_page", gap="small"):
    with st.container(border=True, key="workspace_header"):
        page_header(page.title, descriptions[workspace])
    with st.container(key="workspace_body", gap="small"):
        content, assistant = st.columns([0.62, 0.38], gap="medium")
        with content:
            with st.container(key="workspace_content", height=500, border=False):
                page.run()
                educational_disclaimer()
        with assistant:
            with st.container(key="finpilot_chat_panel", gap="small"):
                conversational_screen(
                    "finance_qa" if workspace == "overview" else workspace, heading="Ask FinPilot"
                )

resizable_chat_divider()
