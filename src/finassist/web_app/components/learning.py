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


# Demo breadth, not a latency guarantee. Keep two narrow definitions first.
QUESTION_GROUPS = (("Simple", 0, 2), ("Medium", 2, 4), ("Semi-medium", 4, 5), ("Complex", 5, 6))

TOPICS = (
    LearningTopic(
        "Investing Basics",
        "Start with one concept, then explore comparisons and multi-part questions.",
        (
            "What is a stock?",
            "What is an ETF?",
            "How does a stock differ from a bond?",
            "What is a fund expense ratio, and why does it matter?",
            "Explain how an ETF trades during the day, with one simple example.",
            "Compare stocks, bonds and ETFs by ownership, income and main risks.",
        ),
    ),
    LearningTopic(
        "Banking & Interest",
        "Begin with interest definitions, then try comparisons and a calculator-backed example.",
        (
            "What is simple interest?",
            "What is compound interest?",
            "How does simple interest differ from compound interest?",
            "Why can APY differ from a savings account's nominal annual rate?",
            "Explain how compounding frequency affects savings, with one simple example.",
            "Compare simple and compound interest on $1,000 at 5% per year for 3 years, "
            "compounded annually for the compound case. Assume no fees or deposits; "
            "show interest and final balances and explain the difference.",
        ),
    ),
    LearningTopic(
        "Bonds & Funds",
        "Explore bond and fund concepts without retirement-account or tax rules.",
        (
            "What is a bond coupon?",
            "What is a mutual fund?",
            "How do bond prices generally respond to rising interest rates?",
            "How does active fund management differ from passive index investing?",
            "Explain ETF market price versus net asset value, with one simple example.",
            "Compare ETFs and mutual funds by trading, pricing and ongoing fees, "
            "and explain why a lower fee alone does not establish lower investment risk.",
        ),
    ),
    LearningTopic(
        "Risk & Diversification",
        "Start with risk vocabulary, then explore the limits of diversification.",
        (
            "What is diversification?",
            "What is concentration risk?",
            "How does diversification reduce company-specific risk?",
            "What is the difference between market risk and company-specific risk?",
            "Explain why owning several stocks in one sector can still be concentrated.",
            "Compare a single-stock portfolio with a diversified stock-and-bond portfolio: "
            "explain company-specific risk, market risk and why losses remain possible, "
            "without recommending an allocation.",
        ),
    ),
    LearningTopic(
        "Tax Education",
        "Explore the bundled US federal tax references, from definitions to multi-topic examples.",
        (
            "What is Form W-2 used for in US federal taxes?",
            "What is a US federal tax deduction?",
            "How does a US federal tax deduction differ from a tax credit?",
            "How do short-term and long-term capital gains differ for US federal taxes?",
            "Explain standard versus itemized deductions for US federal tax year 2026, "
            "with one simple example and no personal recommendation.",
            "For US federal tax year 2026, explain how wages, bank interest and a stock "
            "sale held for 18 months are generally reported. Identify the relevant forms "
            "and distinguish withholding from final tax liability; do not calculate tax owed.",
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
    topics = [topic for topic in TOPICS if (topic.name == "Tax Education") == (page == "tax")]
    names = [topic.name for topic in topics]
    topic_name = st.selectbox(
        "Choose a learning topic",
        names,
        index=0,
        key=f"learning_topic_{page}",
    )
    topic = next(item for item in topics if item.name == topic_name)
    with card(topic.name):
        st.write(topic.introduction)
        st.caption("Choose a suggested question to edit, then send it in the chat panel.")
        for _label, start, end in QUESTION_GROUPS:
            with st.container(horizontal=True, gap="small"):
                for index in range(start, end):
                    question = topic.questions[index]
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
