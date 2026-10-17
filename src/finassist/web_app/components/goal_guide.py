"""Practical goal examples and editable handoffs to the shared assistant."""

import streamlit as st

from finassist.web_app.navigation import open_workspace

GOALS = {
    "Emergency fund": (
        "Set a cash target based on the expenses you want covered and a timeframe. "
        "Use a 0% return assumption to explore savings from contributions alone.",
        "emergency fund",
        12000,
        2000,
        400,
        24,
    ),
    "Home down payment": (
        "Choose the amount you want to set aside and when you expect to need it. "
        "Keep purchase costs separate when defining your target.",
        "home down payment",
        50000,
        10000,
        700,
        48,
    ),
    "Education": (
        "Define the education expense and deadline you want to plan toward. "
        "Account-specific tax rules can be explored separately in Tax Education.",
        "education savings goal",
        20000,
        3000,
        300,
        36,
    ),
    "Travel or a major purchase": (
        "Give a planned expense a clear amount and date, then compare it with what "
        "you can contribute each month.",
        "travel savings goal",
        6000,
        1000,
        250,
        18,
    ),
}


def goal_guide() -> None:
    st.subheader("Turn a goal into a savings plan")
    st.write(
        "Explore how much to save, what your contributions could grow to, "
        "how long a target could take, or whether your deadline is achievable."
    )
    topic = st.selectbox("What are you saving for?", list(GOALS), key="goal_example_topic")
    description, purpose, target, savings, monthly, months = GOALS[topic]
    st.write(description)
    st.caption(
        "Example questions use illustrative amounts. Select one to edit in chat before sending."
    )
    examples = [
        (
            "How much should I save?",
            f"For my {purpose}, how much should I save monthly to reach "
            f"${target:,} in {months} months? I have ${savings:,} saved. Assume 0% annual return.",
        ),
        (
            "What could I accumulate?",
            f"For my {purpose}, what could ${savings:,} saved plus "
            f"${monthly:,} per month grow to in {months} months at an assumed 3% annual return?",
        ),
        (
            "How long will it take?",
            f"How many months would it take to reach my ${target:,} "
            f"{purpose} with ${savings:,} saved and ${monthly:,} monthly contributions, "
            "assuming 0% annual return?",
        ),
        (
            "Can I meet my deadline?",
            f"Can I reach my ${target:,} {purpose} in {months} months "
            f"with ${savings:,} saved and ${monthly:,} per month at 0% annual return? "
            "Show any projected shortfall or surplus.",
        ),
    ]
    with st.container(horizontal=True, gap="small"):
        for index, (label, question) in enumerate(examples):
            if st.button(label, key=f"goal_question_{index}"):
                open_workspace("goals", question)
    st.caption(
        "For your own question, include the target, current savings, monthly contribution "
        "or deadline, and any assumed annual return. The assistant asks for missing "
        "required details; omitted savings and return default to zero."
    )
    with st.expander("What these projections cover"):
        st.write(
            "The assistant uses deterministic calculators for monthly contributions, future "
            "value, time to goal, and feasibility. Projections assume a constant return "
            "and end-of-month contributions; they do not predict investment performance."
        )
        st.write(
            "Inflation, fees and taxes are not automatically included. Goals are not saved "
            "permanently, and this page does not track accounts or recommend investments."
        )
    st.divider()
