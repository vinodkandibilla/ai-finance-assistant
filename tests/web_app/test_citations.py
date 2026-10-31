import pytest

from finassist.web_app.services.citations import label_citations


@pytest.mark.parametrize(
    "url",
    [
        "https://www.bozemandailychronicle.com/wire/news/article.html",
        "https://www.bozemandailychronicle.com/story?x=1&y=2#section",
        "https://www.bozemandailychronicle.com/a_(story)",
    ],
)
@pytest.mark.parametrize("angle", [False, True])
def test_hostname_labels_preserve_exact_destination(url: str, angle: bool) -> None:
    destination = f"<{url}>" if angle else url
    original = f"Reported news. [E2]({destination})"
    assert label_citations(original) == original.replace("[E2]", "[www.bozemandailychronicle.com]")


def test_multiple_domains_and_existing_labels() -> None:
    text = "[E1](<https://a.test/story>) [E23](https://b.test/news) [Read more](https://c.test)"
    assert label_citations(text) == (
        "[a.test](<https://a.test/story>) [b.test](https://b.test/news) [Read more](https://c.test)"
    )


@pytest.mark.parametrize(
    "text",
    [
        "[E2]",
        "[E2](javascript:bad())",
        "[E1](https://u:p@host.test/a)",
        "ordinary text",
        "![E2](https://images.test/a.png)",
    ],
)
def test_non_web_or_non_citation_text_unchanged(text: str) -> None:
    assert label_citations(text) == text


def test_currency_rendering_does_not_enable_math_or_double_escape():
    from finassist.web_app.services.citations import escape_currency

    text = 'Value $110,500, gain $12,000 and loss -$5,000.'
    escaped = escape_currency(text)
    assert escaped == r'Value \$110,500, gain \$12,000 and loss -\$5,000.'
    assert escape_currency(escaped) == escaped
    assert escape_currency('Use $x + y$ for a formula.') == 'Use $x + y$ for a formula.'


def test_currency_escaping_preserves_source_destinations_and_code():
    from finassist.web_app.services.citations import escape_currency

    text = '[Source](https://example.com/$123) `$100` and $200.'
    assert escape_currency(text) == r'[Source](https://example.com/$123) `$100` and \$200.'
