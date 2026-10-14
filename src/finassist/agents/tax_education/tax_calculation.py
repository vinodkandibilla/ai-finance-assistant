"""2026 federal ordinary-income rate-schedule estimate, not a tax-return engine."""

from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

SOURCE_URL = "https://www.irs.gov/irb/2025-45_IRB"
SOURCE_TITLE = "IRS Revenue Procedure 2025-32, section 4.01"
# Versioned schedule from section 4.01, Tables 1–4. QSS shares the MFJ schedule.
THRESHOLDS = {
    "single": (12400, 50400, 105700, 201775, 256225, 640600),
    "mfj": (24800, 100800, 211400, 403550, 512450, 768700),
    "mfs": (12400, 50400, 105700, 201775, 256225, 384350),
    "hoh": (17700, 67450, 105700, 201750, 256200, 640600),
    "qss": (24800, 100800, 211400, 403550, 512450, 768700),
}
RATES = (10, 12, 22, 24, 32, 35, 37)
CENTS = Decimal("0.01")


class TaxInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    taxable_income: str
    filing_status: Literal["single", "mfj", "mfs", "hoh", "qss"]
    income_basis: Literal["taxable_ordinary"]
    tax_year: Literal[2026] = 2026
    currency: Literal["USD"] = "USD"

    @field_validator("taxable_income")
    @classmethod
    def validate_income(cls, raw):
        try:
            value = Decimal(raw)
            if not value.is_finite() or value < 0 or value > Decimal("1e18"):
                raise ValueError("taxable_income must be finite, nonnegative, and at most 1e18")
            with localcontext() as ctx:
                ctx.prec = 50
                return str(value.quantize(CENTS, rounding=ROUND_HALF_UP))
        except DecimalException:
            raise ValueError("taxable_income must be a supported decimal string") from None


def calculate_federal_income_tax(inputs: dict) -> dict:
    try:
        validated = TaxInputs.model_validate(inputs)
    except ValidationError as error:
        return {
            "status": "error",
            "code": "INVALID_INPUT",
            "errors": [
                {"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                for e in error.errors(include_input=False, include_url=False)
            ],
        }
    with localcontext() as ctx:
        ctx.prec = 50
        income = Decimal(validated.taxable_income)
        lower, total = Decimal(0), Decimal(0)
        breakdown = []
        marginal_rate = RATES[0]
        for upper, rate in zip((*THRESHOLDS[validated.filing_status], None), RATES, strict=True):
            if income <= lower:
                break
            portion = min(income, Decimal(upper)) - lower if upper is not None else income - lower
            tax = portion * Decimal(rate) / 100
            total += tax
            marginal_rate = rate
            breakdown.append(
                {
                    "income_over": str(lower),
                    "income_up_to": str(upper) if upper is not None else None,
                    "taxable_amount": str(portion),
                    "rate_percent": rate,
                    "tax": str(tax.quantize(CENTS, rounding=ROUND_HALF_UP)),
                }
            )
            if upper is not None:
                lower = Decimal(upper)
        return {
            "status": "success",
            "calculation_type": "federal_income_tax",
            "validated_inputs": validated.model_dump(),
            "applied_defaults": {
                k: getattr(validated, k) for k in ("tax_year", "currency") if k not in inputs
            },
            "formula": "sum(taxable income in each bracket * bracket rate)",
            "bracket_breakdown": breakdown,
            "marginal_rate_percent": marginal_rate,
            "estimated_federal_income_tax": str(total.quantize(CENTS, rounding=ROUND_HALF_UP)),
            "effective_rate_percent": str(
                (total / income * 100 if income else Decimal(0)).quantize(
                    CENTS, rounding=ROUND_HALF_UP
                )
            ),
            "rounding": {"working_precision": 50, "mode": "ROUND_HALF_UP", "decimal_places": 2},
            "source": {"title": SOURCE_TITLE, "url": SOURCE_URL, "tax_year": 2026},
            "assumptions": [
                "US federal ordinary taxable income only; deductions already reflected in input",
                "Rate-schedule estimate, not the IRS tax-table lookup or a complete return",
                "Before credits, withholding and estimated payments; not the balance due",
                "Excludes state/local and payroll taxes, capital gains, qualified dividends, "
                "AMT and surtaxes",
            ],
        }
