"""Source-first news discovery, separate from the shared conversational assistant."""

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from finassist.web_app.components.chat import ChatState
from finassist.web_app.navigation import open_workspace
from finassist.web_app.services.api_client import ApiError
from finassist.web_app.services.news import articles_from_response, fetch_news


@st.fragment
def news_browser() -> None:
    with st.form("news_discovery"):
        category, period = st.columns(2)
        with category:
            selected_topic = st.selectbox(
                "Topic", ["US Markets", "Technology", "Earnings", "Economy", "Banking"]
            )
        with period:
            window = st.selectbox("Period", ["Today", "Past week", "Past month"], index=1)
        topic = st.text_input(
            "Search topic", max_chars=200, placeholder="Optional: a company, sector, or event"
        )
        state: ChatState | None = st.session_state.get("conversation")
        blocked = bool(state and (state.thread_id or state.failed_request))
        submitted = st.form_submit_button("Search news", disabled=blocked)
    if submitted:
        st.session_state["news_requested_search"] = (topic.strip() or selected_topic, window)
    retry_control = st.empty()
    retry = st.session_state.get("news_search_error")
    if retry and retry_control.button("Retry news search", disabled=blocked):
        submitted = True
    if submitted:
        requested_topic, requested_window = st.session_state["news_requested_search"]
        try:
            with st.spinner("Finding sourced financial news…"):
                response = fetch_news(requested_topic, requested_window)
            articles = articles_from_response(response)
            if not articles:
                st.session_state["news_search_error"] = (
                    response.clarification_question
                    or "No supported article cards were returned. "
                    "Try a narrower topic or another period."
                )
            else:
                st.session_state["news_search_result"] = (
                    articles,
                    requested_topic,
                    requested_window,
                    datetime.now(ZoneInfo("America/Phoenix")).strftime("%b %d, %Y %H:%M %Z"),
                    response.status,
                )
                st.session_state.pop("news_search_error", None)
        except (ApiError, ValueError):
            st.session_state["news_search_error"] = (
                "News is unavailable. Retry or adjust the search."
            )
    error = st.session_state.get("news_search_error")
    if error:
        if not retry:
            retry_control.button("Retry news search", disabled=blocked)
        st.warning(error)
    else:
        retry_control.empty()
    result = st.session_state.get("news_search_result")
    if not result:
        st.info("Choose a topic and period, then search to browse sourced stories.")
        return
    articles, searched_topic, searched_window, retrieved_at, status = result
    st.caption(f"{searched_topic} · {searched_window} · Retrieved {retrieved_at}")
    if error:
        st.caption("Showing previous results; the latest search did not replace them.")
    if status == "insufficient":
        st.info("Partial coverage: retrieved evidence did not cover every requested aspect.")
    st.caption(
        "Publication dates are shown only when supplied by the provider. "
        "No automatic updates; use chat for deeper questions."
    )
    for index, article in enumerate(articles):
        with st.container(border=True):
            if index == 0:
                st.caption("LEAD STORY")
            st.subheader(article.title)
            st.text(article.summary[:600] + ("…" if len(article.summary) > 600 else ""))
            publisher = article.publisher or "Publisher unavailable"
            published = article.published_at or "Publication date unavailable"
            st.caption(f"{publisher} · {published}")
            with st.container(horizontal=True, gap="small"):
                st.link_button("Read article", article.url)
                if st.button("Ask about this", key=f"news_article_ask_{index}"):
                    open_workspace(
                        "news",
                        (
                            f"Explain this financial news article: {article.title}. "
                            f"Source: {article.url}. Reported summary: {article.summary[:1200]} "
                            "Retrieve supporting evidence before answering; distinguish facts "
                            "from possible implications."
                        ),
                    )
