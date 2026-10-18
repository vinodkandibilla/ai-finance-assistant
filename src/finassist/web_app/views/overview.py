"""Welcoming landing page with session-only question handoffs."""

import streamlit as st

from finassist.web_app.components.layout import card
from finassist.web_app.navigation import open_workspace

st.subheader("What would you like to explore?")
st.write("Learn the essentials, explore market context, or take the next step toward a goal.")
st.caption(
    "Ask a question in the chat panel, or choose a workspace below "
    "to start with a suggested question."
)

st.subheader("Six ways to build financial confidence")
with st.container(horizontal=True, gap="small"):
    for title, description, page, example in [
        (
            "Finance Education",
            "Understand investing concepts with clear, grounded explanations.",
            "finance_qa",
            "Explain how ETFs work and the risks I should understand.",
        ),
        (
            "Portfolio Analysis",
            "Explore allocation, diversification, and concentration.",
            "portfolio",
            "Explain how to evaluate portfolio diversification and concentration.",
        ),
        (
            "Market Intelligence",
            "Put market movements and available quotes into context.",
            "market",
            "What is the latest available price of AAPL?",
        ),
        (
            "Goal Planning",
            "Explore the savings steps behind a financial goal.",
            "goals",
            "How much should I save monthly to reach $10,000 in two years?",
        ),
        (
            "Financial News",
            "Explore financial headlines and evidence-backed summaries.",
            "news",
            "Summarize the latest financial-market news with sources.",
        ),
        (
            "Tax Education",
            "Build your understanding of US tax concepts.",
            "tax",
            "Explain the difference between short-term and long-term capital gains.",
        ),
    ]:
        with card(title, width=340):
            st.write(description)
            if st.button(f"Explore {title}", key=f"explore_{page}", width="stretch"):
                open_workspace(page, example)

st.subheader("A glimpse of your workspace")
st.caption("Synthetic demonstration data only. No live prices or connected financial accounts.")
with st.container(horizontal=True, gap="small"):
    with card("Market preview · Demo", width=520):
        st.caption("Fictional instruments and figures · Not live market data")
        st.metric("Demo Market Index", "4,250.00", "+0.8% illustrative change")
        st.write("Demo Technology Index: **1,820.00** · Demo Bond Index: **980.00**")
        st.caption("These invented index values illustrate the interface, not actual performance.")
        if st.button("Explore market context", width="stretch"):
            open_workspace("market", "How should I interpret daily market index changes?")
    with card("Portfolio preview · Demo", width=520):
        st.caption("Synthetic account · No real holdings or account connection")
        st.metric("Example portfolio value", "$100,000")
        st.write("**Stocks 60% · Bonds 30% · Cash 10%**")
        st.caption("Illustrative allocation only; not an investment recommendation.")
        if st.button("Explore portfolio allocation", width="stretch"):
            open_workspace(
                "portfolio", "Explain a hypothetical 60% stocks, 30% bonds, 10% cash mix."
            )
