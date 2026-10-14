"""Direct tax calculator and a thin structured-output adapter."""

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from finassist.agents.tax_education.tax_calculation import calculate_federal_income_tax

CALCULATOR = "calculate_federal_ordinary_income_tax"


@tool(CALCULATOR, response_format="content_and_artifact")
def calculate_federal_ordinary_income_tax(
    taxable_income: str | None = None,
    filing_status: str | None = None,
    income_basis: str | None = None,
    tax_year: int | None = None,
    currency: str | None = None,
    unsupported_features: list[str] | None = None,
) -> tuple[dict, dict]:
    """Estimate US federal ordinary income tax using progressive brackets, not mental math.

    Required: taxable_income as a normalized decimal string, filing_status as
    single/mfj/mfs/hoh/qss, income_basis='taxable_ordinary'. No preparation tool.
    Defaults: tax_year=2026, currency=USD. Only that year/currency are supported.
    Never treat explicit gross income or salary as taxable income, subtract deductions
    twice, or silently ignore missing/ambiguous inputs. Ask all missing details together
    through ask_tax_clarification. Pass unsupported features (credits, payments,
    capital gains, dividends, AMT, state/payroll taxes) until simplification is agreed.
    Returns validated inputs, bracket breakdown, marginal/effective rates, estimated
    federal income tax before credits/payments, defaults, assumptions, and IRS source,
    or structured validation errors. This estimates liability, not the final balance due.
    """
    if unsupported_features:
        result = {
            "status": "error",
            "code": "UNSUPPORTED_FEATURE",
            "errors": [
                {
                    "field": f,
                    "message": "Unsupported; clarify a simplified ordinary-income estimate",
                }
                for f in unsupported_features
            ],
        }
    else:
        inputs = dict(
            taxable_income=taxable_income,
            filing_status=filing_status,
            income_basis=income_basis,
            tax_year=tax_year,
            currency=currency,
        )
        result = calculate_federal_income_tax({k: v for k, v in inputs.items() if v is not None})
    return result, result


def process_tax_calculation_results(result, messages, _evidence):
    calculation = next(
        (
            m.artifact
            for m in reversed(messages)
            if isinstance(m, ToolMessage) and m.name == CALCULATOR and isinstance(m.artifact, dict)
        ),
        None,
    )
    if calculation is None or result.status == "out_of_scope":
        return result, {}, False
    success = calculation["status"] == "success"
    if not success:
        result.status = "insufficient"
        result.answer = "The federal income-tax calculation could not be completed. " + "; ".join(
            f"{e['field']}: {e['message']}" for e in calculation.get("errors", [])
        )
        result.clarification_question = None
    return (
        result,
        {"calculation": {**calculation, "phase": "completed" if success else "error"}},
        success,
    )
