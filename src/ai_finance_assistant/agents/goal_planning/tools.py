"""Deterministic savings projections with end-of-month contributions."""

import math

from langchain_core.tools import tool

ASSUMPTIONS = [
    "Contributions occur at the end of each month; returns compound monthly.",
    "Annual return is a nominal percentage divided by 12; omitted return defaults to 0%.",
    "Constant hypothetical return; excludes taxes, fees, inflation, and withdrawals.",
]


def _calculate(kind: str, **inputs: float) -> dict:
    errors = [
        {"field": name, "message": "Must be finite and nonnegative."}
        for name, value in inputs.items()
        if not math.isfinite(value) or value < 0
    ]
    months = None
    if "time_horizon_years" in inputs:
        years = inputs["time_horizon_years"]
        if math.isfinite(years) and 0 <= years <= 100:
            months = round(years * 12)
            if months <= 0 or not math.isclose(years * 12, months, abs_tol=1e-8):
                errors.append(
                    {
                        "field": "time_horizon_years",
                        "message": "Use a positive horizon in whole months (years = months / 12).",
                    }
                )
        elif math.isfinite(years) and years > 100:
            errors.append({"field": "time_horizon_years", "message": "Maximum is 100 years."})
    if errors:
        return {"status": "error", "code": "INVALID_INPUT", "errors": errors}

    savings = inputs["current_savings"]
    rate = inputs["expected_annual_return"] / 100 / 12
    target = inputs.get("target_amount", 0)
    contribution = inputs.get("monthly_contribution", 0)
    result = {
        "status": "success",
        "calculation_type": kind,
        "inputs": inputs,
        "assumptions": ASSUMPTIONS.copy(),
    }
    try:
        if kind == "time_to_goal":
            if savings >= target:
                duration = 0
            elif contribution == 0 and (rate == 0 or savings == 0):
                return {
                    **result,
                    "reachable": False,
                    "months_to_goal": None,
                    "years_to_goal": None,
                    "message": "The target cannot be reached with these inputs.",
                }
            elif rate == 0:
                duration = math.ceil((target - savings) / contribution)
            else:
                duration = math.ceil(
                    math.log1p((target - savings) / (savings + contribution / rate))
                    / math.log1p(rate)
                )
            return {
                **result,
                "reachable": True,
                "months_to_goal": duration,
                "years_to_goal": round(duration / 12, 2),
            }

        growth = math.exp(months * math.log1p(rate)) if rate else 1.0
        factor = math.expm1(months * math.log1p(rate)) / rate if rate else months
        if kind == "required_contribution":
            required = max(0.0, (target - savings * growth) / factor)
            # Round up so the displayed contribution still funds the target.
            amount = math.ceil(round(required * 100, 8)) / 100
            return {**result, "months": months, "required_monthly_contribution": amount}
        future = savings * growth + contribution * factor
        if not math.isfinite(future):
            raise OverflowError
        result.update(months=months, future_value=round(future, 2))
        if kind == "goal_feasibility":
            result.update(
                feasible=future >= target,
                shortfall=round(max(0.0, target - future), 2),
                surplus=round(max(0.0, future - target), 2),
            )
        return result
    except (OverflowError, ValueError, ZeroDivisionError):
        return {
            "status": "error",
            "code": "CALCULATION_ERROR",
            "errors": [{"field": "inputs", "message": "Projection exceeds supported range."}],
        }


@tool
def calculate_required_contribution(
    target_amount: float,
    time_horizon_years: float,
    current_savings: float = 0,
    expected_annual_return: float = 0,
) -> dict:
    """Calculate monthly savings needed for a target and horizon in years.

    Annual return is a percentage (7 means 7%); defaults to 0%. Savings default to 0.
    Contributions are end-of-month; horizon must equal a whole number of months.
    """
    return _calculate(
        "required_contribution",
        target_amount=target_amount,
        time_horizon_years=time_horizon_years,
        current_savings=current_savings,
        expected_annual_return=expected_annual_return,
    )


@tool
def calculate_future_value(
    monthly_contribution: float,
    time_horizon_years: float,
    current_savings: float = 0,
    expected_annual_return: float = 0,
) -> dict:
    """Project savings from monthly contributions over a horizon in years.

    Annual return is a percentage (7 means 7%); defaults to 0%. Savings default to 0.
    Contributions are end-of-month; horizon must equal a whole number of months.
    """
    return _calculate(
        "future_value",
        monthly_contribution=monthly_contribution,
        time_horizon_years=time_horizon_years,
        current_savings=current_savings,
        expected_annual_return=expected_annual_return,
    )


@tool
def calculate_time_to_goal(
    target_amount: float,
    monthly_contribution: float,
    current_savings: float = 0,
    expected_annual_return: float = 0,
) -> dict:
    """Estimate whole months needed to reach a target with monthly contributions.

    Annual return is a percentage (7 means 7%); defaults to 0%. Savings default to 0.
    Contributions are end-of-month. Reports when a target is unreachable.
    """
    return _calculate(
        "time_to_goal",
        target_amount=target_amount,
        monthly_contribution=monthly_contribution,
        current_savings=current_savings,
        expected_annual_return=expected_annual_return,
    )


@tool
def check_goal_feasibility(
    target_amount: float,
    monthly_contribution: float,
    time_horizon_years: float,
    current_savings: float = 0,
    expected_annual_return: float = 0,
) -> dict:
    """Check if monthly contributions reach a target by the requested horizon in years.

    Returns projected value, feasibility, shortfall and surplus. Annual return is a
    percentage (7 means 7%); defaults to 0%. Savings default to 0. End-of-month deposits.
    """
    return _calculate(
        "goal_feasibility",
        target_amount=target_amount,
        monthly_contribution=monthly_contribution,
        time_horizon_years=time_horizon_years,
        current_savings=current_savings,
        expected_annual_return=expected_annual_return,
    )
