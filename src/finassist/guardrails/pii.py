"""Presidio-based PII redaction before external model calls."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Broad NER labels (organizations, dates, locations, and unqualified names)
# are common in financial education and are not private identifiers by themselves.
SENSITIVE_ENTITIES = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_ITIN",
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    entities: tuple[str, ...]


def redact_pii(text: str) -> RedactionResult:
    """Redact sensitive identifiers and explicitly labeled personal names."""
    try:
        from presidio_anonymizer.entities import OperatorConfig
    except ModuleNotFoundError:
        logger.warning("Presidio is not installed; PII redaction is unavailable")
        return RedactionResult(text=text, entities=())

    # Cover explicit name labels even when the spaCy PERSON model is unavailable.
    person_found = bool(
        re.search(
            r"(?i)\b(?:my\s+name\s+is|name\s*:|customer\s+name\s*:|account\s+holder\s+is)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
            text,
        )
    )
    results = _analyzer().analyze(
        text=text, language="en", entities=list(SENSITIVE_ENTITIES)
    )
    results = [result for result in results if result.entity_type in SENSITIVE_ENTITIES]
    if not results:
        if person_found:
            redacted = re.sub(
                r"(?i)(\b(?:my\s+name\s+is|name\s*:|customer\s+name\s*:|account\s+holder\s+is)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                r"\1[REDACTED_PERSON]",
                text,
            )
            return RedactionResult(text=redacted, entities=("PERSON",))
        return RedactionResult(text=text, entities=())
    anonymized = _anonymizer().anonymize(
        text=text,
        analyzer_results=results,
        operators={
            result.entity_type: OperatorConfig(
                "replace", {"new_value": f"[REDACTED_{result.entity_type}]"}
            )
            for result in results
        },
    )
    redacted_text = re.sub(
        r"(?i)(\b(?:my\s+name\s+is|name\s*:|customer\s+name\s*:|account\s+holder\s+is)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"\1[REDACTED_PERSON]",
        anonymized.text,
    )
    entities = {result.entity_type for result in results}
    if person_found:
        entities.add("PERSON")
    return RedactionResult(text=redacted_text, entities=tuple(sorted(entities)))


@lru_cache(maxsize=1)
def _analyzer():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Presidio's email recognizer uses tldextract. Disable its filesystem and
    # network cache so redaction works in read-only or offline deployments.
    try:
        import tldextract
        extractor = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())
        tldextract.extract = extractor
    except Exception:  # pragma: no cover - optional runtime integration
        pass

    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        return AnalyzerEngine(nlp_engine=provider.create_engine())
    except Exception as exc:  # pragma: no cover - depends on local spaCy runtime
        logger.warning("spaCy model unavailable; using Presidio regex recognizers: %s", exc)
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "no_op",
                "models": [{"lang_code": "en", "model_name": "none"}],
            }
        )
        from presidio_analyzer.context_aware_enhancers import ContextAwareEnhancer

        class PatternOnlyContextEnhancer(ContextAwareEnhancer):
            """Keep pattern scores unchanged when no NLP tokens are available."""

            def enhance_using_context(
                self, text, raw_results, nlp_artifacts, recognizers, context=None
            ):
                return raw_results

        engine = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            context_aware_enhancer=PatternOnlyContextEnhancer(0, 0, 0, 0),
        )
        engine.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="US_SSN",
                patterns=[Pattern("ssn", r"\b\d{3}-\d{2}-\d{4}\b", 0.95)],
                context=["ssn", "social security"],
            )
        )
        return engine


@lru_cache(maxsize=1)
def _anonymizer():
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()
