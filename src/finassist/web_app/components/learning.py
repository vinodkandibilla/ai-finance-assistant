"""Curated learning prompts; all answers come from the existing HTTP chat adapter."""

from dataclasses import dataclass

import streamlit as st

from finassist.web_app.components.chat import ChatState
from finassist.web_app.components.layout import card


@dataclass(frozen=True)
class LearningTopic:
    name: str
    introduction: str
    questions: tuple[str, ...]


TOPICS = (
    LearningTopic(
        "Investing Basics",
        "Start with the vocabulary: stocks, bonds, funds, and investment costs. "
        "Ask for a simple example before comparing choices.",
        (
            "What is the difference between a stock, a bond, and an ETF?",
            "Explain how ETFs work, including fees and risks, for a beginner.",
            "What questions should a beginner ask before researching an investment?",
        ),
    ),
    LearningTopic(
        "Banking & Interest",
        "Explore how account terms, borrowing costs, and interest calculations fit together. "
        "Ask the AI to define unfamiliar terms and show its calculations.",
        (
            "Explain simple interest versus compound interest with a hypothetical example.",
            "What is the difference between APR and APY?",
            "What fees and terms should I understand when comparing savings accounts?",
        ),
    ),
    LearningTopic(
        "Retirement Accounts",
        "Learn the language of retirement accounts: contribution, withdrawal, and tax treatment. "
        "Include your country and tax year when asking about rules or limits.",
        (
            "Explain traditional versus Roth retirement accounts in the US for a beginner.",
            "What is the difference between an IRA and a 401(k)?",
            "What should I ask about employer matching and vesting?",
        ),
    ),
    LearningTopic(
        "Risk & Diversification",
        "Explore different kinds of investment risk and the meaning of diversification. "
        "Ask for scenarios rather than a personalized investment recommendation.",
        (
            "What does diversification mean, and what risks can remain?",
            "Explain concentration risk with a hypothetical portfolio example.",
            "How do time horizon and risk tolerance relate to investing?",
        ),
    ),
    LearningTopic(
        "Tax Education",
        "Explore US tax vocabulary, filing concepts, and investment-related tax questions. "
        "Mention the tax year and jurisdiction; ask for sources and check current rules.",
        (
            "Explain short-term versus long-term capital gains in the US.",
            "What is the difference between a tax deduction and a tax credit?",
            "Explain marginal versus effective tax rates with a hypothetical example.",
        ),
    ),
)


def _select_question(page: str, question: str) -> None:
    st.session_state[f"draft_{page}"] = question
    st.session_state[f"question_{page}"] = question
    st.session_state.pop("clear_chat_draft", None)
    st.session_state["focus_chat_draft"] = page


def learning_hub(page: str) -> None:
    state: ChatState | None = st.session_state.get("conversation")
    blocked = bool(state and (state.thread_id or state.failed_request))
    names = [topic.name for topic in TOPICS]
    topic_name = st.selectbox(
        "Choose a learning topic",
        names,
        index=4 if page == "tax" else 0,
        key=f"learning_topic_{page}",
    )
    topic = next(item for item in TOPICS if item.name == topic_name)
    with card(topic.name):
        st.write(topic.introduction)
        st.caption("Choose a suggested question to edit, then send it in the chat panel.")
        with st.container(horizontal=True, gap="small"):
            for index, question in enumerate(topic.questions):
                st.button(
                    question,
                    key=f"learn_{page}_{topic_name}_{index}",
                    on_click=_select_question,
                    args=(page, question),
                    disabled=blocked,
                )
    if blocked:
        st.caption(
            "Finish the pending reply, resolve the failed request, or choose New Chat "
            "before starting another suggested question."
        )
    st.caption(
        "Follow up in the chat panel. For a new question, include the topic and details it needs; "
        "the API does not receive previous messages unless resuming clarification."
    )
