"""One session-only conversation shared by all six domain workspaces."""

from dataclasses import dataclass, field

import streamlit as st
from pydantic import ValidationError
from streamlit.errors import StreamlitAPIException

from finassist.web_app.components.chat_viewport import fit_chat_viewport
from finassist.web_app.components.focus import focus_chat_composer
from finassist.web_app.config import WebSettings
from finassist.web_app.services.api_client import ApiError, FinanceApiClient
from finassist.web_app.services.citations import label_citations
from finassist.web_app.services.models import QueryRequest, QueryResponse, ResumeRequest
from finassist.web_app.services.showcase import safe_source, scope_market_question

LABELS = {
    "finance_qa": "Finance Education",
    "portfolio": "Portfolio Analysis",
    "market": "Market Intelligence",
    "goals": "Goal Planning",
    "news": "Financial News",
    "tax": "Tax Education",
    "clarify": "Clarification",
    "other": "General",
}


@dataclass
class Message:
    role: str
    text: str
    response: QueryResponse | None = None


@dataclass
class ChatState:
    messages: list[Message] = field(default_factory=list)
    thread_id: str | None = None
    failed_request: QueryRequest | ResumeRequest | None = None
    error: str | None = None
    pending_request: QueryRequest | ResumeRequest | None = None


def _explain_previous_error(state: ChatState, text: str) -> bool:
    """Explain a known application failure without inventing a provider diagnosis."""
    if state.thread_id or text.strip().lower().rstrip("?.! ") not in {
        "why",
        "why did it fail",
        "why the error",
        "what went wrong",
    }:
        return False
    previous = state.messages[-1] if state.messages else None
    if not previous or not previous.response or previous.response.status != "error":
        return False
    state.messages.extend(
        [
            Message("user", text.strip()),
            Message(
                "assistant",
                "The previous request returned a backend error. "
                f"The API reported: {previous.response.answer}\n\n"
                "It did not supply a detailed cause, so I cannot confirm why it failed. "
                "Try the full original question again, including the topic and timeframe "
                "when relevant.",
            ),
        ]
    )
    return True


def _send(state: ChatState, request: QueryRequest | ResumeRequest) -> None:
    st.session_state["scroll_to_latest_turn"] = True
    try:
        with st.spinner("FinPilot is working on your question…"):
            with FinanceApiClient(WebSettings.from_environment()) as client:
                response = (
                    client.resume(request)
                    if isinstance(request, ResumeRequest)
                    else client.query(request)
                )
        if response.clarification_question and not response.thread_id:
            raise ApiError(
                "The API requested clarification without a thread. Retry or start a new chat.",
                code="invalid_response",
            )
    except ApiError as error:
        state.error = str(error)
        state.failed_request = request
        return
    except ValueError:
        state.error = "Check the frontend API configuration before retrying."
        state.failed_request = request
        return
    state.error = None
    state.failed_request = None
    state.messages.append(Message("assistant", response.answer, response))
    state.thread_id = response.thread_id if response.clarification_question else None


def _render_message(message: Message) -> None:
    with st.chat_message(message.role), st.container(gap="xsmall"):
        response = message.response
        if response:
            st.caption(f"FinPilot · {LABELS[response.intent]}")
            if response.classified_intent:
                st.caption(f"Classified intent: {response.classified_intent}")
            if response.executed_handler:
                st.caption(f"Agent: {response.executed_handler}")
        # Assistant answers contain Markdown citations; raw HTML remains disabled.
        if message.text:
            if response:
                st.markdown(label_citations(message.text), unsafe_allow_html=False)
            else:
                st.text(message.text)
        if response:
            if response.status:
                st.text(f"Status: {response.status}")
            if response.status == "error":
                st.warning(
                    "The agent could not complete this request. You can rephrase your question."
                )
            if response.clarification_question:
                st.info("More information is needed. Reply below to continue.")
                st.text(response.clarification_question)
            if response.citations:
                with st.expander("Sources"):
                    for index, citation in enumerate(response.citations, 1):
                        host = safe_source(citation)
                        if host:
                            st.link_button(citation, citation)
                        else:
                            st.text(f"{index}. {citation}")
            else:
                st.caption("No sources were supplied with this response.")
            if response.calculation is not None or response.calculations is not None:
                with st.expander("Calculation details"):
                    if response.calculation is not None:
                        st.json(response.calculation)
                    if response.calculations is not None:
                        st.json(response.calculations)


def _rerun_chat(*, refresh_content: bool = False) -> None:
    if refresh_content:
        st.rerun()
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        # A page handoff can execute the fragment during a full-app run, where
        # Streamlit permits only an app-scoped rerun.
        st.rerun()


def _refresh_matching_content(state: ChatState, page: str) -> bool:
    response = state.messages[-1].response if state.messages else None
    return bool(response and response.intent == page and page in {"portfolio", "news"})


@st.fragment
def conversational_screen(page: str, *, heading: str = "Ask FinPilot") -> None:
    with st.container(key="finpilot_chat", gap="xsmall"):
        _conversational_screen(page, heading=heading)


def _clear_chat() -> None:
    st.session_state["conversation"] = ChatState()
    st.session_state.pop("confirm_chat_reset", None)
    st.session_state["reset_chat_composer"] = True
    st.session_state.pop("queued_question", None)
    for key in list(st.session_state):
        if isinstance(key, str) and key.startswith(
            ("draft_", "question_", "chat_input_", "portfolio_scope_")
        ):
            del st.session_state[key]
    st.rerun()


def _dismiss_new_chat() -> None:
    st.session_state.pop("confirm_chat_reset", None)


@st.dialog("Start a new chat?", on_dismiss=_dismiss_new_chat)
def _confirm_new_chat() -> None:
    st.write("This will clear your current messages and draft. This can’t be undone.")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", key="cancel_new_chat", width="stretch"):
            st.session_state.pop("confirm_chat_reset", None)
            st.rerun()
    with confirm:
        if st.button("Start new chat", key="confirm_new_chat", type="primary", width="stretch"):
            _clear_chat()


def _conversational_screen(page: str, *, heading: str) -> None:
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = ChatState()
    state: ChatState = st.session_state["conversation"]
    with st.container(key="finpilot_chat_header", gap="xsmall"):
        title, action = st.columns([1, 0.15], vertical_alignment="center")
        with title:
            st.subheader(heading)
        with action:
            reset_chat = st.button(
                "",
                icon=":material/edit_square:",
                help="New chat — clears this conversation",
                type="tertiary",
                key="new_chat",
            )
        st.caption(
            "Ask about finance, portfolios, markets, goals, news, or US taxes. "
            "Your chat stays available as you switch pages and isn’t saved permanently."
        )
    if reset_chat:
        has_draft = any(
            bool(st.session_state[key])
            for key in st.session_state
            if isinstance(key, str) and key.startswith(("draft_", "question_"))
        )
        if state.messages or state.thread_id or state.failed_request or has_draft:
            st.session_state["confirm_chat_reset"] = True
        else:
            _clear_chat()
    if st.session_state.get("confirm_chat_reset"):
        _confirm_new_chat()
    latest_question = next(
        (
            index
            for index in range(len(state.messages) - 1, -1, -1)
            if state.messages[index].role == "user"
        ),
        len(state.messages),
    )
    with st.container(key="chat_history", height=300, border=False, autoscroll=False):
        for message in state.messages[:latest_question]:
            _render_message(message)
        with st.container(key=f"latest_chat_turn_{len(state.messages)}", gap="xsmall"):
            for message in state.messages[latest_question:]:
                _render_message(message)
            loading_indicator = st.empty()
            if state.error:
                st.error(state.error)
                st.caption(
                    "Retry resends the same request. "
                    "A timed-out request may already have completed."
                )
                if st.button("Retry last request", key="retry_chat") and state.failed_request:
                    with loading_indicator.container():
                        _send(state, state.failed_request)
                    _rerun_chat(refresh_content=_refresh_matching_content(state, page))
                if st.button("Edit failed message", key="edit_chat") and state.failed_request:
                    failed = state.failed_request
                    edited_text = (
                        failed.reply if isinstance(failed, ResumeRequest) else failed.query
                    )
                    st.session_state[f"draft_{page}"] = edited_text
                    st.session_state[f"question_{page}"] = edited_text
                    state.failed_request = None
                    state.error = None
                    _rerun_chat()
            if state.thread_id:
                st.caption(
                    "Awaiting your clarification reply. Use New Chat to start a different question."
                )
    # Suggested questions and edited failures prefill the native composer once.
    composer_key = f"chat_input_{page}"
    draft = st.session_state.pop(f"question_{page}", None)
    if draft is not None:
        st.session_state[composer_key] = draft
    if st.session_state.pop("reset_chat_composer", False):
        st.session_state[composer_key] = ""
    if page == "portfolio":
        with st.container(key="chat_scope"):
            with st.expander("Optional portfolio request scope"):
                st.caption(
                    "Requested scope only. These fields do not establish identity or grant access."
                )
                st.selectbox(
                    "Requested role",
                    ["API default", "portfolio_manager", "individual_customer"],
                    key="portfolio_scope_role",
                )
                st.text_input(
                    "Customer ID (optional)", max_chars=100, key="portfolio_scope_customer"
                )
                st.text_input("Manager ID (optional)", max_chars=100, key="portfolio_scope_manager")
    with st.container(key="chat_composer", gap="xsmall"):
        text = st.chat_input(
            "Reply to FinPilot…" if state.thread_id else "Ask a financial question…",
            key=composer_key,
            max_chars=4000,
            disabled=state.failed_request is not None or state.pending_request is not None,
        )
    fit_chat_viewport(
        scroll_to_latest=st.session_state.pop("scroll_to_latest_turn", False),
        turn_revision=len(state.messages),
    )
    if st.session_state.get("focus_chat_draft") == page:
        st.session_state.pop("focus_chat_draft", None)
        focus_chat_composer()
    # Render the submitted question and scroll it into view before blocking on HTTP.
    if state.pending_request is not None:
        pending_request = state.pending_request
        state.pending_request = None
        with loading_indicator.container():
            _send(state, pending_request)
        _rerun_chat(refresh_content=_refresh_matching_content(state, page))
    queued = st.session_state.pop("queued_question", None)
    if queued is not None and not state.thread_id and state.failed_request is None:
        text = queued
    if text is not None:
        if _explain_previous_error(state, text):
            st.session_state["scroll_to_latest_turn"] = True
            _rerun_chat()
            return
        try:
            request: QueryRequest | ResumeRequest = (
                ResumeRequest(thread_id=state.thread_id, reply=text)
                if state.thread_id
                else QueryRequest.model_validate(
                    {
                        "query": scope_market_question(text) if page == "market" else text,
                        **(
                            {
                                "user_role": st.session_state.get("portfolio_scope_role")
                                if st.session_state.get("portfolio_scope_role") != "API default"
                                else None,
                                "customer_id": st.session_state.get(
                                    "portfolio_scope_customer", ""
                                ).strip()
                                or None,
                                "manager_id": st.session_state.get(
                                    "portfolio_scope_manager", ""
                                ).strip()
                                or None,
                            }
                            if page == "portfolio"
                            else {}
                        ),
                    }
                )
            )
        except ValidationError:
            st.warning("Enter a question or reply between 1 and 4,000 characters.")
            return
        state.messages.append(Message("user", text.strip()))
        state.pending_request = request
        st.session_state["scroll_to_latest_turn"] = True
        st.session_state.pop(f"draft_{page}", None)
        _rerun_chat()
