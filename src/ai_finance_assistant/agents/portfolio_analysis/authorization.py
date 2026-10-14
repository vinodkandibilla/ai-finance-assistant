"""Simulated application identity; never supplied by model tool arguments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UserContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    role: Literal["PORTFOLIO_MANAGER", "CUSTOMER"]


def customer_id(value: str) -> str:
    """Accept CUST001 examples as aliases for existing C001 dataset identifiers."""
    if value.startswith("CUST") and value[4:].isdigit():
        return "C" + value[4:]
    return value


def denied() -> dict[str, str]:
    return {
        "status": "error",
        "code": "ACCESS_DENIED",
        "message": "You do not have access to the requested portfolio information.",
    }
