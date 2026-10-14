"""Shared validation, rounding, and arithmetic for simple and compound interest."""

from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

USD_QUANTUM = Decimal("0.01")
PERIODS = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12, "daily": 365}


class InterestInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    principal: str
    currency: Literal["USD"] = "USD"
    annual_rate: str
    rate_units: Literal["percent", "decimal"]
    rate_type: Literal["APR", "APY"] = "APR"
    duration_years: str

    @field_validator("principal", "annual_rate", "duration_years")
    @classmethod
    def validate_number(cls, raw, info):
        field = info.field_name
        try:
            value = Decimal(raw)
            if not value.is_finite() or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
            if field == "duration_years" and value == 0:
                raise ValueError("duration_years must be positive")
            if field == "principal":
                with localcontext() as ctx:
                    ctx.prec = 50
                    return str(value.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP))
        except DecimalException:
            raise ValueError(f"{field} must be a supported decimal value") from None
        return raw


class CompoundInterestInputs(InterestInputs):
    compounding_frequency: Literal["annual", "semiannual", "quarterly", "monthly", "daily"] = (
        "annual"
    )


class SimpleInterestInputs(InterestInputs):
    rate_type: Literal["APR"] = "APR"


def calculate_interest(inputs: dict, *, kind: Literal["compound", "simple"]) -> dict:
    """Return a success/error object; never use floating-point arithmetic."""
    unsupported = set(inputs) & {"deposits", "withdrawals", "fees", "taxes"}
    if unsupported:
        return {
            "status": "error",
            "code": "UNSUPPORTED_FEATURE",
            "errors": [
                {"field": field, "message": "Unsupported in single-principal calculation"}
                for field in sorted(unsupported)
            ],
        }
    try:
        validated = (
            SimpleInterestInputs if kind == "simple" else CompoundInterestInputs
        ).model_validate(inputs)
    except ValidationError as error:
        return {
            "status": "error",
            "code": "INVALID_INPUT",
            "errors": [
                {
                    "field": ".".join(map(str, item["loc"])) or "inputs",
                    "code": item["type"],
                    "message": item["msg"],
                }
                for item in error.errors(include_input=False, include_url=False)
            ],
        }
    defaults = {
        name: getattr(validated, name)
        for name in (
            ("currency", "rate_type")
            if kind == "simple"
            else ("currency", "rate_type", "compounding_frequency")
        )
        if name not in inputs
    }
    try:
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.Emax = 18
            principal = Decimal(validated.principal)
            rate = Decimal(validated.annual_rate) / (
                100 if validated.rate_units == "percent" else 1
            )
            duration = Decimal(validated.duration_years)
            periods = PERIODS[validated.compounding_frequency] if kind == "compound" else None
            if kind == "simple":
                balance = principal + principal * rate * duration
                formula = "I = P * r * t; A = P + I"
                substitution = f"I = {principal} * {rate} * {duration}; A = {principal} + I"
            elif validated.rate_type == "APR":
                balance = principal * (1 + rate / periods) ** (periods * duration)
                formula = "A = P * (1 + r/n)^(n*t)"
                substitution = f"{principal} * (1 + {rate}/{periods})^({periods}*{duration})"
            else:
                balance = principal * (1 + rate) ** duration
                formula = "A = P * (1 + y)^t"
                substitution = f"{principal} * (1 + {rate})^{duration}"
            if balance > Decimal("1e18"):
                return {
                    "status": "error",
                    "code": "CALCULATION_LIMIT_EXCEEDED",
                    "errors": [{"field": "final_balance", "message": "Final balance exceeds 1e18"}],
                }
            rounded = balance.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP)
            interest = (rounded - principal).quantize(USD_QUANTUM)
    except DecimalException:
        return {
            "status": "error",
            "code": "CALCULATION_LIMIT_EXCEEDED",
            "errors": [
                {
                    "field": "inputs",
                    "message": "Calculation exceeds supported precision or magnitude",
                }
            ],
        }
    assumptions = [
        "Constant annual rate; hypothetical growth, not guaranteed returns",
        "Single initial principal; no deposits, withdrawals, fees, taxes, or conversion",
        "Principal rounded to USD cents before growth; final balance rounded once",
        "No intermediate rounding",
    ]
    if kind == "simple":
        assumptions.append("Interest is earned only on the original principal; no compounding")
        assumptions.append("Fractional years accrue interest proportionally")
    else:
        assumptions.append("Fractional years use fractional-exponent growth")
    if kind == "compound" and validated.compounding_frequency == "daily":
        assumptions.append("Daily compounding uses 365 equal periods per year")
    if validated.rate_type == "APY":
        assumptions.append(
            "APY already includes compounding; frequency does not change effective annual growth"
        )
    return {
        "status": "success",
        "validated_inputs": validated.model_dump(),
        "normalized_annual_rate": str(rate),
        **({"periods_per_year": periods} if kind == "compound" else {}),
        "calculation_type": kind,
        "applied_defaults": defaults,
        "formula": {"expression": formula, "substitution": substitution},
        "interest_earned": str(interest),
        "final_balance": str(rounded),
        "rounding": {
            "working_precision": 50,
            "currency_decimal_places": 2,
            "mode": "ROUND_HALF_UP",
        },
        "assumptions": assumptions,
    }
