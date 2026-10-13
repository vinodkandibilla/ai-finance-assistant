"""Direct calculator tools and a checkpoint-backed clarification/output adapter."""

from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from ai_finance_assistant.agents.common.rag_tools import finance_citations
from ai_finance_assistant.agents.finance_qa.compound_interest import calculate
from ai_finance_assistant.agents.finance_qa.simple_interest import calculate as calculate_simple
from ai_finance_assistant.rag.models import RetrievedDocument

CALCULATOR = "calculate_compound_interest"
SIMPLE_CALCULATOR = "calculate_simple_interest"
CLARIFICATION = "ask_interest_calculation_clarification"
CALCULATORS = {CALCULATOR, SIMPLE_CALCULATOR}


def calculation_replies(messages):
    return [
        m
        for m in messages
        if isinstance(m, ToolMessage)
        and m.name == CLARIFICATION
        and (m.artifact or {}).get("kind") == "user_reply"
    ]


def create_calculation_tools(*, kind="compound"):
    name = SIMPLE_CALCULATOR if kind == "simple" else CALCULATOR
    compute = calculate_simple if kind == "simple" else calculate

    def execute(inputs, unsupported_features, state, tool_call_id):
        exhausted = any(
            isinstance(m, ToolMessage) and (m.artifact or {}).get("phase") == "insufficient"
            for m in state["messages"]
        )
        if exhausted:
            result = {
                "status": "error",
                "code": "CLARIFICATION_LIMIT_EXCEEDED",
                "errors": [{"field": "inputs", "message": "Clarification budget exhausted"}],
            }
        elif unsupported_features:
            result = {
                "status": "error",
                "code": "UNSUPPORTED_FEATURE",
                "errors": [
                    {"field": feature, "message": "Ask whether to omit this feature"}
                    for feature in unsupported_features
                ],
            }
        else:
            result = compute({key: value for key, value in inputs.items() if value is not None})
        result = {
            **result,
            "calculation_type": kind,
            "calculation_id": tool_call_id,
            "clarification_messages_sent": len(calculation_replies(state["messages"])),
        }
        if result["status"] != "success" and result["clarification_messages_sent"] >= 3:
            result.update(
                phase="insufficient",
                unresolved_inputs=[e.get("field", "inputs") for e in result["errors"]],
            )
        return result, result

    @tool(name, response_format="content_and_artifact")
    def compound(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        principal: str | None = None,
        annual_rate: str | None = None,
        rate_units: str | None = None,
        duration_years: str | None = None,
        currency: str | None = None,
        rate_type: str | None = None,
        compounding_frequency: str | None = None,
        unsupported_features: list[str] | None = None,
    ) -> tuple[dict, dict]:
        """Calculate compound interest directly from user inputs; normalize 100K to '100000'.

        Required: principal, annual_rate, explicit rate_units ('percent' or 'decimal'),
        duration_years. Omitted currency defaults to USD (only supported currency),
        rate_type to APR (APR/APY supported), compounding_frequency to annual
        (annual/semiannual/quarterly/monthly/daily). Do not invent missing or ambiguous inputs.
        No deposits, withdrawals, fees, taxes, conversion or continuous compounding: report
        requested unsupported_features until the user agrees to omit them.
        Returns validated inputs, defaults, formula, interest earned, final balance,
        rounding and assumptions, or structured errors. Clarify all invalid inputs together.
        """
        return execute(
            dict(
                principal=principal,
                annual_rate=annual_rate,
                rate_units=rate_units,
                duration_years=duration_years,
                currency=currency,
                rate_type=rate_type,
                compounding_frequency=compounding_frequency,
            ),
            unsupported_features,
            state,
            tool_call_id,
        )

    @tool(name, response_format="content_and_artifact")
    def simple(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        principal: str | None = None,
        annual_rate: str | None = None,
        rate_units: str | None = None,
        duration_years: str | None = None,
        currency: str | None = None,
        rate_type: str | None = None,
        unsupported_features: list[str] | None = None,
    ) -> tuple[dict, dict]:
        """Calculate simple interest directly from user inputs, without compounding.

        Required: principal, annual_rate, explicit rate_units ('percent' or 'decimal'),
        duration_years. Omitted currency defaults to USD (only supported currency),
        rate_type to nominal APR. APY is unsupported; clarify a nominal annual rate.
        Normalize shorthand amounts, but never invent missing or ambiguous inputs.
        Report requested deposits, withdrawals, fees, taxes or conversion in
        unsupported_features until the user agrees to omit them. Returns validated inputs,
        defaults, formula, interest earned, final balance, rounding and assumptions,
        or structured errors. Clarify all invalid inputs together.
        """
        return execute(
            dict(
                principal=principal,
                annual_rate=annual_rate,
                rate_units=rate_units,
                duration_years=duration_years,
                currency=currency,
                rate_type=rate_type,
            ),
            unsupported_features,
            state,
            tool_call_id,
        )

    return [simple if kind == "simple" else compound]


def create_calculation_clarification_tool():
    @tool(CLARIFICATION, response_format="content_and_artifact")
    def clarify(
        question: str,
        unresolved_inputs: list[str],
        state: Annotated[dict, InjectedState],
        calculation_type: Literal["simple", "compound"] = "compound",
        explanation: str = "",
    ) -> tuple[str, dict]:
        """Ask all missing/ambiguous calculation details together; wait for the user's reply.

        Share one clarification for comparisons. At most three messages per calculation
        request; after the third reply evaluate inputs or tool validation errors and finish
        if still insufficient. explanation may contain a short retrieved handbook explanation.
        """
        count = len(calculation_replies(state["messages"]))
        payload = {
            "calculation_type": calculation_type,
            "unresolved_inputs": unresolved_inputs,
            "clarification_messages_sent": min(count + 1, 3),
            "phase": "collecting",
        }
        if count >= 3:
            payload["phase"] = "insufficient"
            payload["message"] = f"Insufficient details to calculate {calculation_type} interest"
            return payload["message"], payload
        documents = [
            RetrievedDocument.model_validate(d)
            for m in state["messages"]
            if isinstance(m, ToolMessage) and m.name == "finance_rag_query"
            for d in (m.artifact or {}).get("candidates", [])
        ]
        reply = str(
            interrupt(
                {
                    "kind": "clarification",
                    "question": question,
                    "calculation": payload,
                    "answer": explanation if documents else "",
                    "citations": finance_citations(documents),
                }
            )
        )
        return reply, {"kind": "user_reply", **payload}

    return clarify


def create_calculation_result_processor():
    def process(result, messages, evidence):
        artifacts = [
            m.artifact
            for m in messages
            if isinstance(m, ToolMessage) and m.name in CALCULATORS and isinstance(m.artifact, dict)
        ]
        # Keep the latest attempt for each method; a repaired error is no longer a failure.
        latest = {a["calculation_type"]: a for a in artifacts}
        replies = calculation_replies(messages)
        exhausted = next(
            (
                m.artifact
                for m in reversed(messages)
                if isinstance(m, ToolMessage) and (m.artifact or {}).get("phase") == "insufficient"
            ),
            None,
        )
        if result.status == "out_of_scope":
            return result, {}, False
        failures = [a for a in latest.values() if a["status"] != "success"]
        if exhausted or (len(replies) >= 3 and (failures or not latest)):
            pending = exhausted or {**replies[-1].artifact, "phase": "insufficient"}
            unresolved = (
                list(
                    dict.fromkeys(
                        e.get("field", "inputs") for a in failures for e in a.get("errors", [])
                    )
                )
                or pending["unresolved_inputs"]
            )
            pending = {**pending, "unresolved_inputs": unresolved}
            kind = pending["calculation_type"]
            result.status = "insufficient"
            result.answer = (
                f"Insufficient details to calculate {kind} interest\n"
                "Unresolved inputs: " + "; ".join(unresolved)
            )
            result.clarification_question = None
            return result, {"calculation": pending}, False
        if not latest:
            return result, {}, False
        if failures:
            result.status = (
                "error"
                if any(a.get("code") == "CALCULATION_LIMIT_EXCEEDED" for a in failures)
                else "insufficient"
            )
        results = [
            {**a, "phase": "completed" if a["status"] == "success" else "error"}
            for a in latest.values()
        ]
        supplemental = (
            {"calculation": results[0]} if len(results) == 1 else {"calculations": results}
        )
        return result, supplemental, any(a["status"] == "success" for a in results)

    return process
