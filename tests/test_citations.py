"""Tests for src/backend/citations.py: the USED line and scripture reference checks."""

import logging

from src.backend.citations import UsedMarkerFilter, check_references, find_references, split_used

# The Alma 32 test case: the answer cited 32:21 for "desire to believe" (Alma 32:27).
ANSWER = ('Alma compares faith to a seed. He emphasizes that the "desire to believe" is the first step '
          "in allowing the word of God to take root in our hearts (Alma 32:21).")
MANUAL = ("Institute Manual (Book of Mormon), Chapter 30: Alma 32–35 — Alma 32:21. Faith and Hope\n\n"
          "Scriptures: Alma 32:21; Hebrews 11:1\n\nFaith is not a perfect knowledge.")
GOSPEL_TOPICS = ("The prophet Alma taught that the word of God helps strengthen faith. Comparing the word to a "
                 "seed, he said that the “desire to believe” can lead us to “give place” for the word.")


def test_used_line_is_split_off():
    assert split_used("Faith grows.\n\nUSED: [1], [3]") == ("Faith grows.", [1, 3])
    assert split_used("Faith grows.\n**USED:** 2, 4") == ("Faith grows.", [2, 4])
    assert split_used("Faith grows.\nUSED: none") == ("Faith grows.", None)
    assert split_used("Faith grows.") == ("Faith grows.", None)
    # Numbers the model puts in the text count as used, and are taken out.
    assert split_used("Faith grows [2]. Hope too [1, 4].\nUSED: [2]") == ("Faith grows. Hope too.", [2, 1, 4])


def test_stream_filter_never_shows_the_used_line():
    for chunks in (["Faith", " grows.\n\nUSED: [1]"], ["Faith grows.\n\nU", "S", "ED", ": [1], [2]"],
                   list("Faith grows.\nUSED: [1]")):
        stream = UsedMarkerFilter()
        shown = "".join(stream.feed(c) for c in chunks) + stream.finish()
        assert shown.strip() == "Faith grows."


def test_stream_filter_releases_text_that_only_looked_like_the_marker():
    stream = UsedMarkerFilter()
    assert stream.feed("The U") == "The "
    assert stream.feed("S Army") == "US Army"
    stream = UsedMarkerFilter()
    assert stream.feed("Truth: U") == "Truth: "
    assert stream.finish() == "U"


def test_references_are_found_in_every_form():
    found = find_references("Alma 32:27–43; D&C 121:7-8, Doctrine and Covenants 89, 1 Nephi 3:7, Alma 32–35, "
                            "Psalm 23, 1 Ne. 5:10–13, Moro. 10:4, JS—H 1:17, Mosiah 4:27, 34:2")
    assert [(r.book, r.chapter, r.chapter_end, sorted(r.verses) if r.verses else None) for r in found] == [
        ("Alma", 32, None, list(range(27, 44))),
        ("Doctrine and Covenants", 121, None, [7, 8]),
        ("Doctrine and Covenants", 89, None, None),
        ("1 Nephi", 3, None, [7]),
        ("Alma", 32, 35, None),
        ("Psalms", 23, None, None),
        ("1 Nephi", 5, None, [10, 11, 12, 13]),
        ("Moroni", 10, None, [4]),
        ("Joseph Smith—History", 1, None, [17]),
        ("Mosiah", 4, None, [27, 34]),
    ]


def test_a_reference_the_passages_give_is_verified():
    check = check_references("Faith is a hope in things not seen (Alma 32:28).",
                             ["Experimenting on the word (Alma 32:27–43) brings conversion."])
    assert (check.verified, check.total, check.corrections, check.supporting) == (1, 1, [], {0})
    assert check.answer == "Faith is a hope in things not seen (Alma 32:28)."


def test_alma_32_quotation_is_corrected_to_the_verse_its_passage_gives(caplog):
    passage = GOSPEL_TOPICS + " (Alma 32:27)."
    with caplog.at_level(logging.INFO):
        check = check_references(ANSWER, [MANUAL, passage])
    assert check.answer.endswith("(Alma 32:27).")
    assert [(c.original, c.replacement) for c in check.corrections] == [("Alma 32:21", "Alma 32:27")]
    assert check.verified == 0 and check.supporting == {1}
    assert "Alma 32:21 -> Alma 32:27" in caplog.text


def test_alma_32_quotation_without_a_verse_in_its_passage_is_narrowed_to_the_chapter():
    # Alma 32:21 is in the passages, but not for this quotation: the passage
    # that quotes it names Alma and gives no verse, so only "Alma 32" holds.
    check = check_references(ANSWER, [MANUAL, GOSPEL_TOPICS])
    assert check.answer.endswith("(Alma 32).")
    assert check.corrections[0].replacement == "Alma 32"


def test_a_quotation_from_someone_else_does_not_decide_the_reference():
    answer = 'Members say, "I can\'t afford not to pay tithing." In Malachi 3:10, the Lord promises blessings.'
    passages = ["Many say, “I can't afford not to pay tithing.”", "Malachi 3:8–12 promises the windows of heaven."]
    assert check_references(answer, passages).corrections == []


def test_unsupported_references_are_narrowed_or_removed():
    passages = ["Alma the Younger was converted (Mosiah 27).", "Baptism is by immersion."]
    check = check_references(
        "Alma saw an angel (Mosiah 27:11). Jesus was baptized (see Matthew 3:16). "
        "In Matthew 3:16, the heavens opened. He prayed (Alma 36:18; Mosiah 27:11).", passages)
    assert check.answer == ("Alma saw an angel (Mosiah 27). Jesus was baptized. "
                            "In the scriptures, the heavens opened. He prayed (Mosiah 27).")
    assert [c.replacement for c in check.corrections] == ["Mosiah 27", "", "", "", "Mosiah 27"]
    assert check.verified == 0 and check.total == 5


def test_doctrine_and_covenants_sections_named_in_prose_support_the_chapter():
    passage = "Portions of these letters are now found in sections 121, 122, and 123 of the Doctrine and Covenants."
    check = check_references("His letters became D&C 121 and D&C 123.", [passage])
    assert (check.verified, check.total) == (2, 2)
    # A lesson's own "Section 1:" heading isn't a D&C section.
    lesson = "Doctrine and Covenants lesson. Section 1: What troubles did the Saints face?"
    assert check_references("See D&C 1.", [lesson]).verified == 0
