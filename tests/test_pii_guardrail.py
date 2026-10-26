from finassist.guardrails.pii import redact_pii


def test_presidio_redacts_common_financial_pii():
    result = redact_pii(
        "Email alice@example.com, SSN 123-45-6789, phone (602) 555-1212, "
        "card 4111 1111 1111 1111."
    )

    assert "alice@example.com" not in result.text
    assert "123-45-6789" not in result.text
    assert "555-1212" not in result.text
    assert "4111 1111 1111 1111" not in result.text
    assert {"EMAIL_ADDRESS", "US_SSN", "PHONE_NUMBER", "CREDIT_CARD"} <= set(result.entities)
