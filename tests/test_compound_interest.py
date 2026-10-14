from decimal import Decimal

import pytest

from finassist.agents.finance_qa.compound_interest import calculate

BASE = {"principal": "10000", "annual_rate": "6", "rate_units": "percent", "duration_years": "5"}


def test_default_annual_apr_example_and_decimal_rate_equivalence():
    result = calculate(BASE)
    assert result["interest_earned"] == "3382.26"
    assert result["final_balance"] == "13382.26"
    assert result["applied_defaults"] == {
        "currency": "USD",
        "rate_type": "APR",
        "compounding_frequency": "annual",
    }
    assert (
        calculate({**BASE, "annual_rate": "0.06", "rate_units": "decimal"})["final_balance"]
        == result["final_balance"]
    )


def test_monthly_apr_and_apy_are_distinct():
    apr = calculate({**BASE, "compounding_frequency": "monthly"})
    apy = calculate({**BASE, "rate_type": "APY", "compounding_frequency": "monthly"})
    assert Decimal(apr["final_balance"]) > Decimal(apy["final_balance"])
    assert apy["final_balance"] == "13382.26"
    assert "y" in apy["formula"]["expression"]


@pytest.mark.parametrize(
    "change",
    [
        {"principal": "-1"},
        {"principal": "NaN"},
        {"annual_rate": "Infinity"},
        {"duration_years": "0"},
        {"annual_rate": "-2"},
        {"rate_type": "unknown"},
        {"currency": "CAD"},
        {"compounding_frequency": "continuous"},
        {"deposits": "100"},
        {"principal": 10000},
    ],
)
def test_invalid_or_unsupported_inputs_have_no_numerical_results(change):
    result = calculate({**BASE, **change})
    assert result["status"] == "error"
    assert result["errors"]
    assert "final_balance" not in result


@pytest.mark.parametrize("field", ["principal", "annual_rate", "rate_units", "duration_years"])
def test_required_values_are_not_defaulted(field):
    inputs = {k: v for k, v in BASE.items() if k != field}
    assert calculate(inputs)["status"] == "error"


def test_fractional_year_currency_rounding_and_zero_values():
    result = calculate(
        {
            **BASE,
            "principal": "1000",
            "annual_rate": "10",
            "rate_type": "APY",
            "duration_years": "0.5",
        }
    )
    assert result["final_balance"] == "1048.81"
    assert calculate({**BASE, "principal": "0"})["final_balance"] == "0.00"
    assert calculate({**BASE, "annual_rate": "0"})["interest_earned"] == "0.00"


def test_large_output_is_bounded():
    assert (
        calculate({**BASE, "annual_rate": "1000", "duration_years": "100"})["code"]
        == "CALCULATION_LIMIT_EXCEEDED"
    )


def test_all_independently_invalid_numeric_fields_are_reported_together():
    result = calculate({**BASE, "principal": "-1", "annual_rate": "NaN", "duration_years": "0"})
    assert {error["field"] for error in result["errors"]} == {
        "principal",
        "annual_rate",
        "duration_years",
    }


def test_round_half_up_and_daily_periods_are_explicit():
    result = calculate({**BASE, "principal": "1", "annual_rate": "0.5", "duration_years": "1"})
    assert result["final_balance"] == "1.01"
    assert result["interest_earned"] == "0.01"
    assert calculate({**BASE, "compounding_frequency": "daily"})["periods_per_year"] == 365


@pytest.mark.parametrize("currency", ["EUR", "GBP", "JPY", "CAD"])
def test_non_usd_currency_is_rejected_without_conversion(currency):
    result = calculate({**BASE, "currency": currency})
    assert result["status"] == "error"
    assert "final_balance" not in result
    assert any(error["field"] == "currency" for error in result["errors"])


def test_principal_is_rounded_explicitly_instead_of_rejected():
    result = calculate({**BASE, "principal": "100.005", "annual_rate": "0"})
    assert result["status"] == "success"
    assert result["validated_inputs"]["principal"] == "100.01"
    assert result["final_balance"] == "100.01"
    assert result["interest_earned"] == "0.00"


def test_no_arbitrary_duration_or_rate_limit_for_valid_results():
    result = calculate({**BASE, "principal": "1", "annual_rate": "0", "duration_years": "101"})
    assert result["status"] == "success"
    assert result["final_balance"] == "1.00"
    assert (
        calculate({**BASE, "principal": "1", "annual_rate": "2000", "duration_years": "1"})[
            "final_balance"
        ]
        == "21.00"
    )
