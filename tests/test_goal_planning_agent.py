import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ai_finance_assistant.agents.goal_planning import GoalPlanningAgent
from ai_finance_assistant.agents.goal_planning.agent import GoalPlanningResult
from test_finance_autonomous_agent import ScriptedModel, call


class GoalModel(ScriptedModel):
    final: GoalPlanningResult


def make_agent(responses, answer="Tool-supported projection", status="answered", **kwargs):
    model = GoalModel(
        responses=responses,
        final=GoalPlanningResult(status=status, answer=answer, sufficient=True, aligned=True),
    )
    return GoalPlanningAgent(llm=model, **kwargs), model


@pytest.mark.parametrize(
    "name,args,field,value",
    [
        (
            "calculate_required_contribution",
            {"target_amount": 100000, "time_horizon_years": 5},
            "required_monthly_contribution",
            1666.67,
        ),
        (
            "calculate_future_value",
            {"monthly_contribution": 500, "time_horizon_years": 10, "expected_annual_return": 7},
            "future_value",
            86542.4,
        ),
        (
            "calculate_time_to_goal",
            {"target_amount": 100000, "monthly_contribution": 1000},
            "months_to_goal",
            100,
        ),
        (
            "check_goal_feasibility",
            {"target_amount": 100000, "monthly_contribution": 1000, "time_horizon_years": 5},
            "feasible",
            False,
        ),
    ],
)
async def test_model_selects_each_tool(name, args, field, value):
    agent, model = make_agent([call(name, args, "calc"), AIMessage(content="Done")])
    # The same input can invoke any tool: Python does not route by query keywords.
    result = await agent.ainvoke({"query": "Help me plan my financial goal."})
    assert result["status"] == "answered"
    assert result["intent"] == "goals"
    assert result["calculation"][field] == value
    assert result["answer"] == "Tool-supported projection"
    assert any(isinstance(m, ToolMessage) and m.name == name for m in model.seen[-1])


async def test_clarification_resumes_with_user_reply():
    agent, model = make_agent(
        [
            call("ask_goal_clarification", {"question": "What target and timeline?"}, "ask"),
            call(
                "calculate_required_contribution",
                {"target_amount": 100000, "time_horizon_years": 5},
                "calc",
            ),
            AIMessage(content="Done"),
        ],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "goal-resume"}}
    paused = await agent.ainvoke({"query": "How much should I save?"}, config)
    assert paused["__interrupt__"][0].value == {
        "kind": "clarification",
        "question": "What target and timeline?",
    }
    result = await agent.ainvoke(Command(resume="$100,000 in 5 years"), config)
    assert result["status"] == "answered"
    assert any(
        isinstance(m, ToolMessage) and m.content == "$100,000 in 5 years" for m in model.seen[-1]
    )


async def test_agent_can_call_another_tool_for_complete_answer():
    agent, _ = make_agent(
        [
            call(
                "check_goal_feasibility",
                {"target_amount": 100000, "monthly_contribution": 1000, "time_horizon_years": 5},
                "check",
            ),
            call(
                "calculate_required_contribution",
                {"target_amount": 100000, "time_horizon_years": 5},
                "required",
            ),
            AIMessage(content="Done"),
        ]
    )
    result = await agent.ainvoke({"query": "Can I reach my goal, and what should I save?"})
    assert result["status"] == "answered"
    assert len(result["calculations"]) == 2


async def test_validation_error_can_be_repaired_by_model():
    agent, model = make_agent(
        [
            call(
                "calculate_future_value",
                {"monthly_contribution": -500, "time_horizon_years": 10},
                "bad",
            ),
            call(
                "calculate_future_value",
                {"monthly_contribution": 500, "time_horizon_years": 10},
                "good",
            ),
            AIMessage(content="Done"),
        ]
    )
    result = await agent.ainvoke({"query": "Invest $500 per month for ten years."})
    assert result["status"] == "answered"
    assert result["calculation"]["future_value"] == 60000
    assert any(isinstance(m, ToolMessage) and "INVALID_INPUT" in m.content for m in model.seen[-1])


@pytest.mark.parametrize(
    "responses",
    [
        [AIMessage(content="Invented projection")],
        [
            call(
                "calculate_future_value",
                {"monthly_contribution": -1, "time_horizon_years": 5},
                "bad",
            ),
            AIMessage(content="Done"),
        ],
        [
            call("calculate_future_value", {"monthly_contribution": 500}, "missing"),
            AIMessage(content="Done"),
        ],
    ],
)
async def test_answer_requires_successful_calculator(responses):
    agent, _ = make_agent(responses)
    result = await agent.ainvoke({"query": "Project my savings"})
    assert result["status"] == "insufficient"
    assert "Tool-supported projection" not in result["answer"]


async def test_out_of_scope_does_not_require_calculation():
    agent, _ = make_agent([AIMessage(content="Submit a new question")], status="out_of_scope")
    result = await agent.ainvoke({"query": "Give me a stock quote"})
    assert result["status"] == "out_of_scope"
    assert "calculation" not in result


async def test_timeout_returns_safe_error():
    agent, _ = make_agent([], timeout_seconds=0.01)

    async def slow(_):
        await asyncio.sleep(1)

    agent.agent = RunnableLambda(slow)
    result = await agent.ainvoke({"query": "Project my savings"})
    assert result["status"] == "error"
    assert result["errors"] == ["TimeoutError"]
    assert "calculation" not in result


@pytest.mark.parametrize("stage", ["model", "formatter"])
async def test_provider_failures_return_safe_error_without_partial_calculation(stage):
    agent, _ = make_agent(
        [
            call(
                "calculate_future_value",
                {"monthly_contribution": 500, "time_horizon_years": 5},
                "calc",
            ),
            AIMessage(content="Done"),
        ]
    )

    async def fail(_):
        raise RuntimeError("Provider diagnostics containing private request data")

    if stage == "model":
        agent.agent = RunnableLambda(fail)
    else:
        agent._formatter = RunnableLambda(fail)
    result = await agent.ainvoke({"query": "Project my savings"})
    assert result["status"] == "error"
    assert result["errors"] == ["RuntimeError"]
    assert "private request data" not in str(result)
    assert "calculation" not in result


async def test_invalid_structured_assessment_returns_safe_error():
    agent, _ = make_agent(
        [
            call(
                "calculate_future_value",
                {"monthly_contribution": 500, "time_horizon_years": 5},
                "calc",
            ),
            AIMessage(content="Done"),
        ]
    )
    agent._formatter = RunnableLambda(lambda _: {"status": "answered", "answer": "Incomplete"})
    result = await agent.ainvoke({"query": "Project my savings"})
    assert result["status"] == "error"
    assert result["errors"] == ["ValidationError"]
    assert "calculation" not in result


async def test_college_goal_retains_original_context_after_clarification():
    agent, model = make_agent(
        [
            call(
                "ask_goal_clarification",
                {"question": "When would you like to reach the $50,000 goal?"},
                "ask",
            ),
            call(
                "calculate_required_contribution",
                {"target_amount": 50000, "time_horizon_years": 3},
                "calc",
            ),
            AIMessage(content="Done"),
        ],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "college-goal"}}
    original = "I want to save $50,000 for my daughter's college."
    paused = await agent.ainvoke({"query": original}, config)
    assert paused["__interrupt__"][0].value["question"].startswith("When")
    assert not any(isinstance(m, ToolMessage) for m in model.seen[-1])
    result = await agent.ainvoke(Command(resume="In 3 years."), config)
    assert result["calculation"]["required_monthly_contribution"] == 1388.89
    assert any(m.content == original for m in model.seen[-1])
    assert any(isinstance(m, ToolMessage) and m.content == "In 3 years." for m in model.seen[-1])


@pytest.mark.parametrize(
    "status,sufficient,aligned",
    [
        ("answered", False, True),
        ("answered", True, False),
        ("insufficient", False, True),
    ],
)
async def test_failed_coverage_check_returns_to_tools(status, sufficient, aligned):
    agent, model = make_agent(
        [
            call(
                "check_goal_feasibility",
                {"target_amount": 100000, "monthly_contribution": 1000, "time_horizon_years": 5},
                "first",
            ),
            AIMessage(content="Partial result"),
            call(
                "calculate_required_contribution",
                {"target_amount": 100000, "time_horizon_years": 5},
                "second",
            ),
            AIMessage(content="Complete result"),
        ]
    )
    assessments = [
        GoalPlanningResult(
            status=status,
            sufficient=sufficient,
            aligned=aligned,
            next_action="Calculate the required monthly contribution.",
        ),
        GoalPlanningResult(
            status="answered",
            sufficient=True,
            aligned=True,
            answer="You need $1,666.67 per month under the zero-return assumption.",
        ),
    ]
    agent._formatter = RunnableLambda(lambda _: assessments.pop(0))
    result = await agent.ainvoke(
        {
            "query": "Can I reach $100,000 in 5 years with $1,000 "
            "monthly, and what contribution would I need?"
        }
    )
    assert result["status"] == "answered"
    assert len(result["calculations"]) == 2
    assert any(
        isinstance(m, ToolMessage) and m.name == "check_goal_feasibility" for m in model.seen[-1]
    )


async def test_persistently_insufficient_check_has_bounded_loop():
    agent, _ = make_agent([AIMessage(content="Incomplete") for _ in range(3)])
    agent._formatter = RunnableLambda(
        lambda _: GoalPlanningResult(
            status="answered", sufficient=False, aligned=False, next_action="Clarify the goal."
        )
    )
    result = await agent.ainvoke({"query": "Plan my savings"})
    assert result["status"] == "insufficient"


async def test_hard_stop_blocks_fourth_model_and_tool_turn():
    responses = [
        call(
            "calculate_future_value", {"monthly_contribution": -1, "time_horizon_years": 5}, str(i)
        )
        for i in range(4)
    ]
    agent, model = make_agent(responses)
    result = await agent.ainvoke({"query": "Project my savings"})
    assert len(model.seen) == 3
    assert len(model.responses) == 1
    assert model.responses[0].tool_calls[0]["id"] == "3"
    assert result["status"] == "insufficient"
    assert result["errors"] == ["goal_iteration_limit"]
    assert "three iterations" in result["answer"]
    assert "new request" in result["answer"]


async def test_successful_third_tool_turn_can_complete_request():
    agent, model = make_agent(
        [
            call(
                "calculate_future_value",
                {"monthly_contribution": -1, "time_horizon_years": 5},
                "bad1",
            ),
            call(
                "calculate_future_value",
                {"monthly_contribution": -1, "time_horizon_years": 5},
                "bad2",
            ),
            call(
                "calculate_future_value",
                {"monthly_contribution": 500, "time_horizon_years": 5},
                "good",
            ),
            AIMessage(content="Fourth turn must not run"),
        ]
    )
    result = await agent.ainvoke({"query": "Project $500 monthly over 5 years"})
    assert len(model.seen) == 3
    assert len(model.responses) == 1
    assert result["status"] == "answered"
    assert result["calculation"]["future_value"] == 30000


async def test_iteration_budget_survives_multiple_clarification_replies():
    agent, model = make_agent(
        [
            call("ask_goal_clarification", {"question": "What target?"}, "ask1"),
            call("ask_goal_clarification", {"question": "What deadline?"}, "ask2"),
            call("ask_goal_clarification", {"question": "Please clarify the deadline."}, "ask3"),
            call(
                "calculate_required_contribution",
                {"target_amount": 50000, "time_horizon_years": 3},
                "blocked",
            ),
        ],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "three-clarifications"}}
    await agent.ainvoke({"query": "Plan my savings"}, config)
    await agent.ainvoke(Command(resume="$50,000"), config)
    await agent.ainvoke(Command(resume="Soon"), config)
    result = await agent.ainvoke(Command(resume="3 years"), config)
    assert len(model.seen) == 3
    assert len(model.responses) == 1
    assert result["status"] == "insufficient"
    assert result["errors"] == ["goal_iteration_limit"]
    assert "calculation" not in result
