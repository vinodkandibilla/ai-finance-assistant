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


def test_financial_education_preserves_broad_ner_labels(monkeypatch):
    from types import SimpleNamespace

    from finassist.guardrails import pii

    text = "An ETF is an exchange-traded fund traded throughout the trading day."

    class Analyzer:
        def analyze(self, **kwargs):
            assert set(kwargs["entities"]) == set(pii.SENSITIVE_ENTITIES)
            # Even an unexpected broad label must not reach the anonymizer.
            return [
                SimpleNamespace(entity_type="ORGANIZATION"),
                SimpleNamespace(entity_type="DATE_TIME"),
                SimpleNamespace(entity_type="PERSON"),
            ]

    monkeypatch.setattr(pii, "_analyzer", lambda: Analyzer())
    result = pii.redact_pii(text)
    assert result.text == text
    assert result.entities == ()


def test_explicit_personal_name_still_redacted(monkeypatch):
    from types import SimpleNamespace

    from finassist.guardrails import pii

    monkeypatch.setattr(
        pii, "_analyzer", lambda: SimpleNamespace(analyze=lambda **kwargs: [])
    )
    result = pii.redact_pii("My name is Alice Smith.")
    assert result.text == "My name is [REDACTED_PERSON]."
    assert result.entities == ("PERSON",)


def test_regex_fallback_skips_nlp_context_and_keeps_identifiers(monkeypatch, caplog):
    import logging

    from presidio_analyzer.nlp_engine import NlpEngineProvider

    from finassist.guardrails import pii

    original = NlpEngineProvider.create_engine

    def create_engine(provider):
        if provider.nlp_configuration["nlp_engine_name"] == "spacy":
            raise RuntimeError("NLP runtime unavailable")
        return original(provider)

    monkeypatch.setattr(NlpEngineProvider, "create_engine", create_engine)
    pii._analyzer.cache_clear()
    try:
        with caplog.at_level(logging.INFO):
            result = pii.redact_pii("Email alice@example.com, SSN 123-45-6789.")
        assert "alice@example.com" not in result.text
        assert "123-45-6789" not in result.text
        assert "Skipping context extraction" not in caplog.text
        assert "using Presidio regex recognizers" in caplog.text
    finally:
        pii._analyzer.cache_clear()
