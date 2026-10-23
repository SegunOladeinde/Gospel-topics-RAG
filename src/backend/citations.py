"""Which passages an answer used, and whether its scripture references hold up.

The model is given numbered passages ([1], [2], ...) and ends its answer
with a line "USED: [1], [3]" naming the ones it drew on. UsedMarkerFilter
keeps that line out of the streamed text; split_used() reads it from the
finished answer, so only the sources actually used are shown and saved.

check_references() then checks every scripture reference in the answer
(e.g. "Alma 32:27") against the passages:
- verified when a passage cites it (a range like "Alma 32:27–43" covers
  "Alma 32:28"). When the sentence quotes a passage, the reference must be
  the one that passage gives next to the quotation: "Alma 32:21" isn't
  verified for a quotation of Alma 32:27 just because Alma 32:21 is
  mentioned somewhere else.
- otherwise corrected to the reference the passage shows for that quotation
  or wording, narrowed to the chapter when only the chapter is supported
  ("Alma 32"), or removed. Every change is logged.
"""

import logging
import re
from dataclasses import dataclass, field

from src.backend.ingestion.sections import BOOKS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The USED line
# ---------------------------------------------------------------------------

USED_MARKER = "USED:"
# Usually a line of its own, but found mid-line too (as UsedMarkerFilter does).
_USED_LINE = re.compile(r"[ \t*_]*USED:[ \t*_]*(?P<list>[^\n]*)\s*$")
_INLINE_MARKERS = re.compile(r"[ \t]?\[(\d+(?:\s*,\s*\d+)*)\]")


def split_used(text: str) -> tuple[str, list[int] | None]:
    """(answer without the USED line or inline [n] markers, the passage numbers
    used). None when the model named none (no USED line, or "USED: none")."""
    used: list[int] = []
    match = _USED_LINE.search(text)
    if match:
        used = [int(n) for n in re.findall(r"\d+", match.group("list"))]
        text = text[: match.start()]
    # The model is told not to, but numbers it puts in the text count as used too.
    for inline in _INLINE_MARKERS.finditer(text):
        used += [int(n) for n in re.findall(r"\d+", inline.group(1))]
    text = _INLINE_MARKERS.sub("", text).rstrip()
    return text, list(dict.fromkeys(used)) or None


class UsedMarkerFilter:
    """Passes streamed text through, minus the final "USED: ..." line.

    Text that could be the start of the marker ("U", "US", ...) is held back
    until the next chunk shows whether it is; from the marker on, nothing
    more is passed through.
    """

    def __init__(self):
        self._held = ""
        self._stopped = False

    def feed(self, text: str) -> str:
        if self._stopped:
            return ""
        text = self._held + text
        self._held = ""
        start = text.find(USED_MARKER)
        if start != -1:
            self._stopped = True
            return text[:start].rstrip(" *_")
        for size in range(min(len(USED_MARKER) - 1, len(text)), 0, -1):
            if USED_MARKER.startswith(text[-size:]):
                self._held = text[-size:]
                return text[:-size]
        return text

    def finish(self) -> str:
        """What's still held back once the stream has ended (it wasn't the marker)."""
        held, self._held = ("" if self._stopped else self._held), ""
        return held


# ---------------------------------------------------------------------------
# Scripture references
# ---------------------------------------------------------------------------

_ALIASES = {
    "D&C": "Doctrine and Covenants",
    "Doctrine & Covenants": "Doctrine and Covenants",
    "Psalm": "Psalms",
    "Joseph Smith–History": "Joseph Smith—History",
    "Joseph Smith-History": "Joseph Smith—History",
    "Joseph Smith–Matthew": "Joseph Smith—Matthew",
    "Joseph Smith-Matthew": "Joseph Smith—Matthew",
    "JS—H": "Joseph Smith—History",
    "JS—M": "Joseph Smith—Matthew",
    "A of F": "Articles of Faith",
    "W of M": "Words of Mormon",
}
# The Bible Dictionary and Topical Guide write references abbreviated ("1 Ne. 5:13").
_ABBREVIATIONS = {
    "Gen.": "Genesis", "Ex.": "Exodus", "Lev.": "Leviticus", "Num.": "Numbers", "Deut.": "Deuteronomy",
    "Josh.": "Joshua", "Judg.": "Judges", "1 Sam.": "1 Samuel", "2 Sam.": "2 Samuel", "1 Kgs.": "1 Kings",
    "2 Kgs.": "2 Kings", "1 Chr.": "1 Chronicles", "2 Chr.": "2 Chronicles", "Neh.": "Nehemiah",
    "Esth.": "Esther", "Ps.": "Psalms", "Prov.": "Proverbs", "Eccl.": "Ecclesiastes", "Isa.": "Isaiah",
    "Jer.": "Jeremiah", "Lam.": "Lamentations", "Ezek.": "Ezekiel", "Dan.": "Daniel", "Obad.": "Obadiah",
    "Hab.": "Habakkuk", "Zeph.": "Zephaniah", "Hag.": "Haggai", "Zech.": "Zechariah", "Mal.": "Malachi",
    "Matt.": "Matthew", "Rom.": "Romans", "1 Cor.": "1 Corinthians", "2 Cor.": "2 Corinthians",
    "Gal.": "Galatians", "Eph.": "Ephesians", "Philip.": "Philippians", "Col.": "Colossians",
    "1 Thes.": "1 Thessalonians", "2 Thes.": "2 Thessalonians", "1 Tim.": "1 Timothy", "2 Tim.": "2 Timothy",
    "Philem.": "Philemon", "Heb.": "Hebrews", "1 Pet.": "1 Peter", "2 Pet.": "2 Peter", "1 Jn.": "1 John",
    "2 Jn.": "2 John", "3 Jn.": "3 John", "Rev.": "Revelation", "1 Ne.": "1 Nephi", "2 Ne.": "2 Nephi",
    "3 Ne.": "3 Nephi", "4 Ne.": "4 Nephi", "Hel.": "Helaman", "Morm.": "Mormon", "Moro.": "Moroni",
    "Abr.": "Abraham",
}
_ALIASES |= _ABBREVIATIONS
_BOOK_NAMES = sorted(
    {name for name in BOOKS.values() if not name.startswith("Testimony")} | set(_ALIASES),
    key=len, reverse=True,
)
_VERSES = r"\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*"
# Spaces in book names match any whitespace: the pages use no-break spaces ("1 Ne.").
_BOOK_PATTERN = "|".join(re.escape(b).replace(r"\ ", r"\s+") for b in _BOOK_NAMES)
_REFERENCE = re.compile(
    rf"(?<![\w&])(?P<book>{_BOOK_PATTERN})\s+(?P<chapter>\d+)"
    rf"(?:(?::\s*(?P<verses>{_VERSES}))|(?:\s*[–-]\s*(?P<chapter_end>\d+)(?![\d:])))?"
)
# "sections 121, 122, and 123" in a passage that is about the Doctrine and Covenants;
# lowercase and without a colon, so not a lesson's own "Section 1:" headings.
_DC_SECTIONS = re.compile(r"\bsections?\s+(\d+(?:\s*(?:,\s*and|,|and|&|–|-)\s*\d+)*)(?!\s*:)")
_QUOTE = re.compile(r"[“\"]([^”\"]{8,}?)[”\"]")
_WORD = re.compile(r"[a-z]+")
_STOPWORDS = set("""
    about above after again against also among because been before being below between both
    could does doing down during each from further have having here into itself just more most
    other ought ours over same should some such than that their theirs them then there these
    they this those through under until very were what when where which while whom will with
    would your yours lord jesus christ shall unto thee thou""".split())
_NEARBY = 400  # characters around a quotation in which its reference is looked for


@dataclass(frozen=True)
class Reference:
    book: str  # canonical name, e.g. "Doctrine and Covenants"
    chapter: int
    chapter_end: int | None  # "Alma 32–35"
    verses: frozenset[int] | None  # None: the whole chapter
    text: str  # as written
    start: int
    end: int

    def covers(self, other: "Reference") -> bool:
        """Whether this reference (in a passage) supports `other` (in the answer)."""
        if self.book != other.book:
            return False
        if other.verses is None:  # a chapter: any mention of the chapter will do
            return self.chapter <= other.chapter <= (self.chapter_end or self.chapter)
        return (self.chapter_end is None and self.chapter == other.chapter
                and self.verses is not None and other.verses <= self.verses)

    def chapter_text(self) -> str:
        """"Alma 32:21" -> "Alma 32", keeping the book as written."""
        return self.text[: self.text.index(str(self.chapter), len(self.text.split()[0])) + len(str(self.chapter))]


def _verse_set(verses: str) -> frozenset[int]:
    numbers: set[int] = set()
    for part in verses.split(","):
        bounds = [int(n) for n in re.findall(r"\d+", part)]
        if len(bounds) == 2 and bounds[0] <= bounds[1] and bounds[1] - bounds[0] < 200:
            numbers.update(range(bounds[0], bounds[1] + 1))
        else:
            numbers.update(bounds)
    return frozenset(numbers)


def find_references(text: str) -> list[Reference]:
    references = []
    for m in _REFERENCE.finditer(text):
        written = " ".join(m.group("book").split())
        book = _ALIASES.get(written, written)
        references.append(Reference(
            book=book,
            chapter=int(m.group("chapter")),
            chapter_end=int(m.group("chapter_end")) if m.group("chapter_end") else None,
            verses=_verse_set(m.group("verses")) if m.group("verses") else None,
            text=m.group(0).rstrip(),
            start=m.start(),
            end=m.start() + len(m.group(0).rstrip()),
        ))
    return references


def _passage_references(passage: str) -> list[Reference]:
    references = find_references(passage)
    if "Doctrine and Covenants" in passage or "D&C" in passage:
        for m in _DC_SECTIONS.finditer(passage):
            for number in re.findall(r"\d+", m.group(1)):
                references.append(Reference("Doctrine and Covenants", int(number), None, None,
                                            m.group(0), m.start(), m.end()))
    return references


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", re.sub(r"[\[\]]", "", text))


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 5 and w not in _STOPWORDS}


def _claim(answer: str, reference: Reference) -> str:
    """The sentence a reference supports, without the reference: the one it's
    in ("In Alma 32:21, Alma teaches ..."), or, for a reference standing on its
    own after a sentence ("... hearts. (Alma 32:21)"), the one before."""
    before, after = answer[: reference.start], answer[reference.end:]
    boundaries = [m.end() for m in re.finditer(r"[.!?]\s+", before)]
    start = boundaries[-1] if boundaries else 0
    following = re.search(r"[.!?](\s|$)", after)
    end = reference.end + (following.end() if following else len(after))
    lead = re.sub(r"[(\s]|\bsee\b", "", before[start:], flags=re.IGNORECASE)
    if not lead and not _WORD.search(answer[reference.end:end]) and len(boundaries) > 1:
        start, end = boundaries[-2], reference.start
    return answer[start: reference.start] + " " + answer[reference.end: end]


@dataclass
class Correction:
    original: str
    replacement: str  # "" when removed
    reason: str


@dataclass
class ReferenceCheck:
    answer: str  # with corrections applied
    corrections: list[Correction] = field(default_factory=list)
    verified: int = 0  # references in the answer as given that the passages support
    total: int = 0
    supporting: set[int] = field(default_factory=set)  # passages that supported a reference


def _remove(answer: str, reference: Reference) -> str:
    """The answer without the reference, tidying "(see X)", "(X; Y)" and "in X,"."""
    before, after = answer[: reference.start], answer[reference.end:]
    paren = re.search(r"\(\s*(?:see\s+(?:also\s+)?)?$", before, re.IGNORECASE)
    if paren and re.match(r"\s*\)", after):  # "(see Alma 32:21)" -> ""
        closing = after.index(")") + 1
        return before[: paren.start()].rstrip(" ") + after[closing:]
    if re.match(r"\s*[;,]\s*", after) and paren:  # "(Alma 32:21; Moroni 10:4)" -> "(Moroni 10:4)"
        return before + re.sub(r"^\s*[;,]\s*", "", after)
    if re.search(r"[;,]\s*$", before):  # "(Moroni 10:4; Alma 32:21)" -> "(Moroni 10:4)"
        return re.sub(r"\s*[;,]\s*$", "", before) + after
    return before + "the scriptures" + after


def check_references(answer: str, passages: list[str]) -> ReferenceCheck:
    """Verify, correct or remove each scripture reference in the answer (see the module docstring)."""
    evidence = [(i, ref) for i, passage in enumerate(passages) for ref in _passage_references(passage)]
    normalised = [_normalise(p) for p in passages]
    result = ReferenceCheck(answer=answer)
    edits: list[tuple[Reference, str, str]] = []  # (reference, replacement, reason), applied last

    for reference in find_references(answer):
        result.total += 1
        claim = _claim(answer, reference)
        supporting = [(i, ref) for i, ref in evidence if ref.covers(reference)]

        # A quotation ties the claim to one place in one passage: the reference
        # must be the one given there.
        needle, places = "", []
        for quote in _QUOTE.findall(claim):
            needle = _normalise(quote).strip(" .,;:!?'\"")
            places = [(i, text.find(needle)) for i, text in enumerate(normalised)
                      if len(needle) >= 8 and needle in text]
            if places:
                break
        # Only a quotation its passage attributes to this book ("The prophet Alma
        # taught ... 'desire to believe'") says which verse the claim is about.
        book_words = _book_words(reference)
        places = [(i, position) for i, position in places
                  if any(word in normalised[i][max(0, position - _NEARBY): position + _NEARBY] for word in book_words)]
        if places:
            # Positions are in the normalised text, which is at most a little
            # shorter than the passage: close enough for "nearby".
            nearby = [(abs(ref.start - position), j, ref) for i, position in places for j, ref in evidence
                      if j == i and ref.book == reference.book and ref.verses is not None
                      and abs(ref.start - position) <= _NEARBY]
            covering = [j for _, j, ref in nearby if ref.covers(reference)]
            if covering:
                result.verified += 1
                result.supporting.update(covering)
                continue
            if nearby:
                _, j, best = min(nearby, key=lambda found: found[0])
                edits.append((reference, best.text, f'passage [{j + 1}] gives {best.text} for "{needle}"'))
                result.supporting.add(j)
                continue
            supporting = [(j, ref) for j, ref in supporting if needle in normalised[j]]
            if not supporting:
                edits.append(_fallback(reference, evidence,
                                       f'the passage quoting "{needle}" gives no reference next to it'))
                continue

        if supporting:
            result.verified += 1
            result.supporting.update(i for i, _ in supporting)
            continue

        # Not found: is the same wording next to another verse of this chapter?
        words = _content_words(claim)
        best_score, best = 0, None
        for i, ref in evidence:
            if ref.book != reference.book or ref.chapter != reference.chapter or ref.verses is None:
                continue
            window = passages[i][max(0, ref.start - _NEARBY): ref.end + _NEARBY]
            score = len(words & _content_words(window))
            if score > best_score:
                best_score, best = score, (i, ref)
        if best and best_score >= 3:
            i, ref = best
            edits.append((reference, ref.text, f"passage [{i + 1}] gives {ref.text} for the same wording"))
            result.supporting.add(i)
            continue
        edits.append(_fallback(reference, evidence, "no passage gives it"))

    for reference, replacement, reason in sorted(edits, key=lambda edit: edit[0].start, reverse=True):
        if replacement:
            result.answer = result.answer[: reference.start] + replacement + result.answer[reference.end:]
        else:
            result.answer = _remove(result.answer, reference)
        result.corrections.append(Correction(reference.text, replacement, reason))
        logger.info("Reference %s -> %s (%s)", reference.text, replacement or "removed", reason)
    result.corrections.reverse()  # in reading order
    return result


def _book_words(reference: Reference) -> tuple[str, ...]:
    """How a passage might name the reference's book near a quotation."""
    if reference.book == "Doctrine and Covenants":
        return ("doctrine and covenants", "d&c", "revelation")
    name = reference.book.lower()
    return (name, name.split()[-1]) if name[0].isdigit() else (name,)


def _fallback(reference: Reference, evidence: list[tuple[int, Reference]], reason: str) -> tuple:
    """Narrow a verse reference to its chapter when a passage supports the
    chapter, otherwise remove it."""
    if reference.verses is not None:
        chapter = Reference(reference.book, reference.chapter, None, None, "", 0, 0)
        if any(ref.covers(chapter) for _, ref in evidence):
            return reference, reference.chapter_text(), f"{reason}; only the chapter is supported"
    return reference, "", reason
