"""Tests for src/backend/source_links.py (source id -> label + URL)."""

import pytest

from src.backend.main import QueryResponse
from src.backend.source_links import source_link

SITE = "https://www.churchofjesuschrist.org/study"


@pytest.mark.parametrize(
    ("source_id", "label", "url"),
    [
        (
            "bible_dictionary/baptism",
            "Bible Dictionary: Baptism",
            f"{SITE}/scriptures/bd/baptism?lang=eng",
        ),
        (
            "topical_guide/faith",
            "Topical Guide: Faith",
            f"{SITE}/scriptures/tg/faith?lang=eng",
        ),
        (
            "topics_and_questions/baptism",
            "Gospel Topics: Baptism",
            f"{SITE}/manual/gospel-topics/baptism?lang=eng",
        ),
        (
            "topics_and_questions/baptism-study-guide",
            "Gospel Topics: Baptism (Study Guide)",
            f"{SITE}/manual/gospel-topics/baptism-study-guide?lang=eng",
        ),
    ],
)
def test_known_collections_map_to_absolute_urls(source_id, label, url):
    link = source_link(source_id)
    assert link.id == source_id
    assert link.label == label
    assert link.url == url


@pytest.mark.parametrize("source_id", ["unknown/faith", "faith", "", "/faith", "TOPICAL_GUIDE/faith"])
def test_unknown_collection_is_plain_text(source_id):
    link = source_link(source_id)
    assert link.url is None
    assert link.label == source_id


def test_multi_word_slug_keeps_small_words_lowercase():
    link = source_link("topical_guide/aaron-descendants-of")
    assert link.label == "Topical Guide: Aaron Descendants of"
    assert link.url == f"{SITE}/scriptures/tg/aaron-descendants-of?lang=eng"


def test_numeric_slug():
    assert source_link("bible_dictionary/1-kings").url == f"{SITE}/scriptures/bd/1-kings?lang=eng"


def test_uppercase_and_whitespace_are_normalised():
    link = source_link(" topical_guide/Faith ")
    assert link.url == f"{SITE}/scriptures/tg/faith?lang=eng"
    assert link.label == "Topical Guide: Faith"


@pytest.mark.parametrize(
    "source_id",
    [
        "topical_guide/",  # empty slug
        "topical_guide/faith_hope",  # "_" left by describe_source's sanitising
        "topical_guide/faith--hope",  # double hyphen
        "topical_guide/-faith",  # leading hyphen
        "topical_guide/faith/extra",  # nested path
        "topical_guide/faith?x=1",  # query injection
        "topical_guide/café",  # non-ASCII
    ],
)
def test_malformed_slug_is_plain_text(source_id):
    link = source_link(source_id)
    assert link.url is None
    assert link.label  # still shows something readable


def test_bare_study_guide_suffix_is_not_treated_as_study_guide():
    link = source_link("topics_and_questions/study-guide")
    assert link.label == "Gospel Topics: Study Guide"


def test_query_response_includes_source_links():
    body = QueryResponse(answer="a", sources=["topical_guide/faith", "mystery/x"]).model_dump()
    assert body["sources"] == ["topical_guide/faith", "mystery/x"]
    assert body["source_links"] == [
        {
            "id": "topical_guide/faith",
            "label": "Topical Guide: Faith",
            "url": f"{SITE}/scriptures/tg/faith?lang=eng",
        },
        {"id": "mystery/x", "label": "mystery/x", "url": None},
    ]


def test_prompt_passages_are_labelled_with_readable_names():
    """The model sees "Topical Guide: Faith", so its citations read naturally."""
    from langchain_core.documents import Document

    from src.backend.main import format_context

    text = format_context([Document(page_content="Faith is...", metadata={"source": "topical_guide", "topic": "faith"})])
    assert text.startswith("[Topical Guide: Faith]\n")
    assert "topical_guide/faith" not in text


def test_system_prompt_forbids_mentioning_the_context():
    from src.backend.main import SYSTEM_PROMPT

    assert "never mention" in SYSTEM_PROMPT
    for phrase in ('"context"', '"documents"', '"sources provided"'):
        assert phrase in SYSTEM_PROMPT
