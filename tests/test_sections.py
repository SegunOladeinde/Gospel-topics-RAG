"""Tests for sectioned collections: the Institute manuals and Church History Topics.

Pages are cut into sections (src/backend/ingestion/sections.py) and every
section is ingested as its own document with a title, a deep link and its
scripture references (documents.section_documents).
"""

from langchain_core.documents import Document

import src.backend.ingestion.pipeline as pipeline
from src.backend.ingestion.config import CollectionConfig, IngestConfig
from src.backend.ingestion.documents import load_documents, section_title
from src.backend.ingestion.pipeline import chunk_documents, run_ingestion
from src.backend.ingestion.sections import extract_sections, parse_raw, render_raw, scripture_ref, section_url
from src.backend.main import format_context
from src.backend.source_links import source_link, source_link_for

PAGE = """
<html><body><article id="main">
  <p class="reference">“Chapter 30: Alma 32–35,” Book of Mormon Student Manual (2018)</p>
  <header><p class="title-number" id="title_number1">Chapter 30</p><h1 id="title1">Alma 32–35</h1></header>
  <div class="body">
    <p id="p0">Alma taught the humble Zoramites.<a class="note-ref" href="#note1"><sup class="marker">1</sup></a></p>
    <h2 id="title2">Introduction</h2>
    <p id="p1">Faith grows like a seed (see <a class="scripture-ref" href="/study/scriptures/bofm/alma/32?lang=eng&amp;id=p28-p43#p28">Alma 32:28–43</a>).</p>
    <h2 id="title3">Commentary</h2>
    <h3 id="title8"><a class="scripture-ref" href="/study/scriptures/bofm/alma/32?lang=eng&amp;id=p21#p21">Alma 32:21</a>. Faith and Hope</h3>
    <p id="p2">Faith is not a perfect knowledge (<a href="/study/scriptures/nt/heb/11?lang=eng&amp;id=p1#p1">Hebrews 11:1</a>;
       <a href="/study/scriptures/dc-testament/dc/38?lang=eng&amp;id=p24#p24">D&amp;C 38:24</a>;
       <a href="/study/scriptures/bofm/alma/32?lang=eng&amp;id=p21#p21">Alma 32:21</a>).</p>
    <figure><p class="credit">Painting by Simon Dewey, © IRI</p></figure>
    <aside><p class="title">Church Resources</p><p>A reading list.</p></aside>
    <h4>A small heading kept as text</h4>
    <ul><li><p>Plant the seed.</p></li></ul>
  </div>
  <footer class="notes"><p id="note1_p1">A footnote.</p></footer>
</article></body></html>
"""

URL = "https://www.churchofjesuschrist.org/study/manual/book-of-mormon-student-manual-2018/chapter-30-alma-32-35?lang=eng"


def manual(tmp_path, name="institute_bom", label="Institute Manual (Book of Mormon)", size=1000, overlap=200):
    return CollectionConfig(
        name=name,
        label=label,
        raw_path=tmp_path / name,
        url_pattern=f"https://www.churchofjesuschrist.org/{name}/{{slug}}?lang=eng",
        sections=True,
        chunk_size=size,
        chunk_overlap=overlap,
    )


def save_page(config, slug="chapter-30-alma-32-35", html=PAGE, url=URL, split=("h2", "h3")):
    title, sections = extract_sections(html, split)
    config.raw_path.mkdir(parents=True, exist_ok=True)
    (config.raw_path / f"{slug}.txt").write_text(render_raw(title, url, sections), encoding="utf-8")


def test_scripture_refs_from_links():
    assert scripture_ref("/study/scriptures/bofm/alma/32?lang=eng&id=p21#p21") == "Alma 32:21"
    assert scripture_ref("/study/scriptures/bofm/alma/32?lang=eng&id=p8-p16#p8") == "Alma 32:8–16"
    assert scripture_ref("/study/scriptures/bofm/alma/32?lang=eng&id=p14,16#p14") == "Alma 32:14, 16"
    assert scripture_ref("/study/scriptures/dc-testament/dc/89?lang=eng") == "D&C 89"
    assert scripture_ref("/study/scriptures/pgp/js-h/1?lang=eng&id=p34-p35#p34") == "Joseph Smith—History 1:34–35"
    assert scripture_ref("/study/scriptures/bofm/three?lang=eng") == "Testimony of Three Witnesses"
    assert scripture_ref("/study/scriptures/bofm/alma/32?lang=eng&id=title1") == "Alma 32"
    assert scripture_ref("/study/manual/gospel-topics/faith?lang=eng") is None
    assert scripture_ref("/study/scriptures/bofm/unknown-book/1") is None


def test_page_is_cut_at_its_headings_without_notes_or_credits():
    title, sections = extract_sections(PAGE, ("h2", "h3"))
    assert title == "Chapter 30: Alma 32–35"
    assert [(s.heading, s.anchor) for s in sections] == [
        ("", "title1"), ("Introduction", "title2"), ("Alma 32:21. Faith and Hope", "title8"),
    ]  # "Commentary" has no text of its own
    assert sections[0].text == "Alma taught the humble Zoramites."  # no footnote marker
    assert sections[2].scriptures == ["Alma 32:21", "Hebrews 11:1", "D&C 38:24"]
    text = sections[2].text
    assert "A small heading kept as text" in text and "Plant the seed." in text
    assert "Simon Dewey" not in text and "reading list" not in text and "footnote" not in text
    assert "Book of Mormon Student Manual (2018)" not in " ".join(s.text for s in sections)


def test_raw_file_round_trips():
    title, sections = extract_sections(PAGE, ("h2", "h3"))
    parsed_title, url, parsed = parse_raw(render_raw(title, URL, sections))
    assert (parsed_title, url) == (title, URL)
    assert [(s.heading, s.anchor, s.scriptures, s.text) for s in parsed] == [
        (s.heading, s.anchor, s.scriptures, s.text) for s in sections
    ]


def test_section_urls_deep_link_to_the_heading():
    assert section_url(URL, "title8", first=False) == f"{URL}&id=title8#title8"
    assert section_url(URL, "title1", first=True) == URL
    assert section_url("https://x/page", "title2", first=False) == "https://x/page?id=title2#title2"


def test_section_titles():
    chapter = "Chapter 30: Alma 32–35"
    assert section_title(chapter, "") == chapter
    assert section_title(chapter, "Alma 32:21. Faith and Hope") == "Alma 32:21. Faith and Hope"
    assert section_title(chapter, "1 Nephi 3–4. Obtaining the Plates") == "1 Nephi 3–4. Obtaining the Plates"
    assert section_title(chapter, "D&C 89. The Word of Wisdom") == "D&C 89. The Word of Wisdom"
    assert section_title(chapter, "Introduction") == "Chapter 30: Alma 32–35 — Introduction"
    lesson = "Lesson 5 Class Preparation Material: The Coming Forth of the Book of Mormon"
    assert section_title(lesson, "") == "Lesson 5: The Coming Forth of the Book of Mormon"
    assert section_title(lesson, "Section 2: Witnesses?") == "Lesson 5, Section 2: Witnesses?"
    assert section_title("Liberty Jail", "Aftermath") == "Liberty Jail — Aftermath"


def test_each_section_is_a_document_with_title_link_and_scriptures(tmp_path):
    config = manual(tmp_path)
    save_page(config)
    docs = load_documents(config)
    assert [d.slug for d in docs] == [
        "chapter-30-alma-32-35",
        "chapter-30-alma-32-35--introduction",
        "chapter-30-alma-32-35--alma-32-21-faith-and-hope",
    ]
    faith = docs[2]
    assert faith.title == "Alma 32:21. Faith and Hope"
    assert faith.url == f"{URL}&id=title8#title8"
    assert docs[0].url == URL
    assert faith.metadata == {
        "section": "Alma 32:21. Faith and Hope", "anchor": "title8",
        "scriptures": ["Alma 32:21", "Hebrews 11:1", "D&C 38:24"],
    }
    assert faith.text.startswith("Scriptures: Alma 32:21; Hebrews 11:1; D&C 38:24\n\n")
    assert faith.chunk_prefix == "Institute Manual (Book of Mormon), Chapter 30: Alma 32–35 — Alma 32:21. Faith and Hope"
    assert len({d.sha256 for d in docs}) == 3


def test_every_passage_carries_its_section_prefix(tmp_path):
    config = manual(tmp_path, size=300, overlap=50)
    long_page = PAGE.replace("Faith is not a perfect knowledge", "Faith is not a perfect knowledge. " * 40)
    save_page(config, html=long_page)
    docs = load_documents(config)
    chunks = [c for c in chunk_documents(docs, {config.name: config}) if c.document.slug.endswith("faith-and-hope")]
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.content.startswith(docs[2].chunk_prefix + "\n\n")
        assert len(chunk.content) <= 300


def test_unsectioned_documents_chunk_exactly_as_before(tmp_path):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    config = CollectionConfig(name="topical_guide", label="Topical Guide", raw_path=tmp_path / "tg",
                              url_pattern="https://x/{slug}", chunk_size=200, chunk_overlap=20)
    config.raw_path.mkdir()
    body = "Faith in Jesus Christ. " * 60
    (config.raw_path / "faith.txt").write_text(f"Faith\nSource: https://x/faith\n\n{body}\n", encoding="utf-8")
    doc = load_documents(config)[0]
    expected = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_text(doc.text)
    assert [c.content for c in chunk_documents([doc], {config.name: config})] == expected


def test_passage_rows_carry_section_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "count_tokens", lambda texts, tokenizer: 0)
    config = manual(tmp_path)
    save_page(config)
    rows = []

    class Store:
        def existing_documents(self, collections):
            return {}

        def write_batch(self, documents, sha256, counts, rows_for):
            rows.extend(rows_for({doc.key: i for i, doc in enumerate(documents)}))

    run_ingestion(IngestConfig([config], "m", "t", 0.02), load_documents(config), Store(),
                  embed=lambda texts: [[0.0] for _ in texts], dry_run=False)
    faith = [r for r in rows if "Faith and Hope" in r["content"]]
    assert faith and faith[0]["metadata"]["scriptures"] == ["Alma 32:21", "Hebrews 11:1", "D&C 38:24"]
    assert all(r["metadata"]["anchor"] for r in rows)


def test_source_links_for_the_new_collections():
    link = source_link_for(
        "institute_bom/chapter-30-alma-32-35--alma-32-21-faith-and-hope",
        "Alma 32:21. Faith and Hope", f"{URL}&id=title8#title8",
    )
    assert link.label == "Institute Manual (Book of Mormon): Alma 32:21. Faith and Hope"
    assert link.url == f"{URL}&id=title8#title8"
    # Without a stored URL, the fallback goes to the page itself.
    fallback = source_link("institute_bom/chapter-30-alma-32-35--alma-32-21-faith-and-hope")
    assert fallback.url == URL
    assert source_link("church_history_topics/liberty-jail").label == "Church History Topics: Liberty Jail"
    assert source_link("institute_restoration/lesson-5-x").label.startswith("Institute Manual (Restoration): ")


def test_context_labels_use_the_stored_title():
    doc = Document(page_content="Faith is not a perfect knowledge.", metadata={
        "source": "institute_bom", "topic": "chapter-30-alma-32-35--alma-32-21-faith-and-hope",
        "title": "Alma 32:21. Faith and Hope", "url": f"{URL}&id=title8#title8",
    })
    assert format_context([doc]).startswith("[1] Institute Manual (Book of Mormon): Alma 32:21. Faith and Hope\n")
