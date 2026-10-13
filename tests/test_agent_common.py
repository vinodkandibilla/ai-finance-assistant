from langchain_core.messages import ToolMessage

from ai_finance_assistant.agents.common.runtime import user_replies
from ai_finance_assistant.agents.common.schemas import EvidenceResult


def test_clarification_metadata_distinguishes_user_replies_from_policy_guidance():
    text = "Do not ask for year confirmation alone. This is my actual reply."
    messages = [
        ToolMessage(
            content="Use corpus defaults",
            tool_call_id="policy",
            name="ask_tax_clarification",
            artifact={"kind": "policy_guidance"},
        ),
        ToolMessage(
            content=text,
            tool_call_id="user",
            name="ask_tax_clarification",
            artifact={"kind": "user_reply"},
        ),
        ToolMessage(
            content="Other domain reply",
            tool_call_id="other",
            name="ask_finance_clarification",
            artifact={"kind": "user_reply"},
        ),
    ]
    assert user_replies(messages, "ask_tax_clarification") == [text]


def test_shared_evidence_result_preserves_citation_and_source_metadata():
    result = EvidenceResult.model_validate(
        {
            "candidates": [
                {"content": "Evidence", "source": "corpus", "metadata": {"page_number": 9}}
            ],
            "metadata_filters": {"tax_year": 2026},
            "citations": ["Source"],
            "errors": [],
        }
    )
    assert result.candidates[0].metadata["page_number"] == 9
    assert result.model_dump()["metadata_filters"] == {"tax_year": 2026}
