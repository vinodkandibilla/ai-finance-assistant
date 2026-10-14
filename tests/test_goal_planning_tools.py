import pytest

from finassist.agents.goal_planning.tools import (
    calculate_future_value,
    calculate_required_contribution,
    calculate_time_to_goal,
    check_goal_feasibility,
)


@pytest.mark.parametrize(
    "calculator,inputs",
    [
        (calculate_required_contribution, {"target_amount": 100000, "time_horizon_years": 5}),
        (
            calculate_future_value,
            {"monthly_contribution": 500, "time_horizon_years": 10, "expected_annual_return": 7},
        ),
        (calculate_time_to_goal, {"target_amount": 100000, "monthly_contribution": 1000}),
        (
            check_goal_feasibility,
            {"target_amount": 100000, "monthly_contribution": 1000, "time_horizon_years": 5},
        ),
    ],
)
def test_primary_calculations_are_repeatable(calculator, inputs):
    first = calculator.invoke(inputs)
    assert first["status"] == "success"
    assert calculator.invoke(inputs) == first


def test_required_contribution_without_return():
    result = calculate_required_contribution.invoke(
        {"target_amount": 100000, "time_horizon_years": 5}
    )
    assert result["status"] == "success"
    assert result["required_monthly_contribution"] == 1666.67


def test_future_value_with_return():
    result = calculate_future_value.invoke(
        {"monthly_contribution": 500, "time_horizon_years": 10, "expected_annual_return": 7}
    )
    assert result["future_value"] == 86542.40


def test_time_to_goal_without_return():
    result = calculate_time_to_goal.invoke({"target_amount": 100000, "monthly_contribution": 1000})
    assert result["months_to_goal"] == 100
    assert result["years_to_goal"] == 8.33


def test_goal_feasibility_without_return():
    result = check_goal_feasibility.invoke(
        {"target_amount": 100000, "monthly_contribution": 1000, "time_horizon_years": 5}
    )
    assert result["feasible"] is False
    assert result["future_value"] == 60000
    assert result["shortfall"] == 40000


def test_existing_savings_and_reachable_goal():
    result = check_goal_feasibility.invoke(
        {
            "target_amount": 100000,
            "monthly_contribution": 1000,
            "time_horizon_years": 5,
            "current_savings": 40000,
        }
    )
    assert result["feasible"] is True
    assert result["shortfall"] == result["surplus"] == 0


def test_required_contribution_with_growth_funds_target():
    inputs = {"time_horizon_years": 5, "current_savings": 10000, "expected_annual_return": 7}
    required = calculate_required_contribution.invoke({"target_amount": 100000, **inputs})
    future = calculate_future_value.invoke(
        {"monthly_contribution": required["required_monthly_contribution"], **inputs}
    )
    assert future["future_value"] >= 100000


def test_time_to_goal_rounds_up_to_first_month_reaching_target():
    inputs = {"monthly_contribution": 500, "current_savings": 10000, "expected_annual_return": 7}
    result = calculate_time_to_goal.invoke({"target_amount": 100000, **inputs})
    months = result["months_to_goal"]
    before = calculate_future_value.invoke({"time_horizon_years": (months - 1) / 12, **inputs})
    after = calculate_future_value.invoke({"time_horizon_years": months / 12, **inputs})
    assert before["future_value"] < 100000 <= after["future_value"]


def test_goal_already_reached():
    inputs = {"target_amount": 10000, "current_savings": 10000}
    assert (
        calculate_time_to_goal.invoke({"monthly_contribution": 0, **inputs})["months_to_goal"] == 0
    )
    assert (
        calculate_required_contribution.invoke({"time_horizon_years": 1, **inputs})[
            "required_monthly_contribution"
        ]
        == 0
    )


def test_unreachable_goal():
    result = calculate_time_to_goal.invoke({"target_amount": 100000, "monthly_contribution": 0})
    assert result["reachable"] is False
    assert result["months_to_goal"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("monthly_contribution", -1),
        ("current_savings", -1),
        ("expected_annual_return", -1),
        ("expected_annual_return", float("nan")),
        ("monthly_contribution", float("inf")),
        ("time_horizon_years", 0),
        ("time_horizon_years", 0.01),
        ("time_horizon_years", 101),
    ],
)
def test_invalid_inputs_return_structured_errors(field, value):
    result = calculate_future_value.invoke(
        {"monthly_contribution": 500, "time_horizon_years": 10, field: value}
    )
    assert result["status"] == "error"
    assert result["code"] == "INVALID_INPUT"
    assert any(error["field"] == field for error in result["errors"])


def test_negative_target_is_rejected():
    result = calculate_time_to_goal.invoke({"target_amount": -1, "monthly_contribution": 100})
    assert result["code"] == "INVALID_INPUT"


def test_tool_call_produces_langchain_tool_message():
    from langchain_core.messages import ToolMessage

    result = calculate_future_value.invoke(
        {
            "name": "calculate_future_value",
            "args": {
                "monthly_contribution": 500,
                "time_horizon_years": 1,
            },
            "id": "goal-1",
            "type": "tool_call",
        }
    )
    assert isinstance(result, ToolMessage)
    assert result.name == "calculate_future_value"
