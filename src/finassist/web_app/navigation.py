"""Session-only question handoff between the overview and domain pages."""

import streamlit as st


def open_workspace(page: str, question: str) -> None:
    st.session_state[f"draft_{page}"] = question.strip()
    st.session_state[f"question_{page}"] = question.strip()
    st.switch_page(f"views/{page}.py")
