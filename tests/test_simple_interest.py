import pytest

from ai_finance_assistant.agents.finance_qa.simple_interest import calculate

BASE = {"principal": "10000", "annual_rate": "6", "rate_units": "percent", "duration_years": "5"}


def test_simple_interest_formula_defaults_and_decimal_rate():
    result = calculate(BASE)
    assert result["interest_earned"] == "3000.00"
    assert result["final_balance"] == "13000.00"
    assert result["formula"]["expression"] == "I = P * r * t; A = P + I"
    assert result["calculation_type"] == "simple"
    assert result["applied_defaults"] == {"currency": "USD", "rate_type": "APR"}
    assert "compounding_frequency" not in result["validated_inputs"]
    assert "periods_per_year" not in result
    assert (
        calculate({**BASE, "annual_rate": "0.06", "rate_units": "decimal"})["final_balance"]
        == "13000.00"
    )


def test_fractional_duration_and_rounding():
    result = calculate({**BASE, "principal": "1000", "annual_rate": "10", "duration_years": "0.5"})
    assert result["final_balance"] == "1050.00"
    assert result["interest_earned"] == "50.00"
    assert calculate({**BASE, "principal": "0"})["final_balance"] == "0.00"
    assert calculate({**BASE, "annual_rate": "0"})["interest_earned"] == "0.00"
    assert (
        calculate({**BASE, "principal": "100.005", "annual_rate": "0"})["final_balance"] == "100.01"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"rate_type": "APY"},
        {"currency": "EUR"},
        {"principal": "-1"},
        {"annual_rate": "NaN"},
        {"duration_years": "0"},
        {"compounding_frequency": "monthly"},
        {"fees": "1"},
    ],
)
def test_invalid_inputs_are_not_silently_reinterpreted(change):
    result = calculate({**BASE, **change})
    assert result["status"] == "error"
    assert "final_balance" not in result


@pytest.mark.parametrize("field", ["principal", "annual_rate", "rate_units", "duration_years"])
def test_required_inputs_are_not_defaulted(field):
    assert (
        calculate({key: value for key, value in BASE.items() if key != field})["status"] == "error"
    )
