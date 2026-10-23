"""Checks every claim in an answer against the passages it used.

After the answer is written (and its scripture references checked, see
citations.py), one call to a small model looks at each sentence of the
answer next to the passages the answer drew on, and labels it:

    S  supported: everything it states is in the passages
    N  no claim: a greeting, transition, invitation to pray, the reminder
       that this isn't an official Church resource, a suggested question
    U  unsupported: something in it isn't in the passages; the model gives
       the sentence rewritten to say only what the passages support, or ""
       when nothing in it is

Unsupported sentences are replaced or removed (each change logged). When
more than half the claims are unsupported, the whole answer is replaced by
the "couldn't find this" reply. An answer whose claims all passed is marked
verified, which the UI shows as "Verified against sources".

The check never blocks an answer: if the call fails or times out, the
answer is returned as written, unverified.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the sources I have."
CHECK_TIMEOUT_SECONDS = 8.0

CHECK_PROMPT = """You check an answer about Latter-day Saint teachings against the reference passages it was based on.

For each numbered sentence of the answer, decide:
S - supported: every fact in it (names, dates, numbers, places, events, quotations, scripture
    references, teachings) is stated in the passages or follows directly from them.
N - no factual claim: a greeting, thanks, transition, encouragement, an invitation to pray or
    ponder, a reminder that personal prayer and the Holy Spirit witness truth, a note that this
    is not an official Church resource, a suggested question, or a statement that the sources
    don't cover something.
U - unsupported: anything in it is not in the passages, or goes beyond or contradicts them.

For each U sentence, give "fix": the sentence rewritten to state only what the passages support,
keeping its wording and tone as far as possible, or "" if nothing in it is supported.
Each passage starts with its source's name. Naming a listed source ("Gospel Topics explains",
"(Institute Manual: 1 Nephi 4:10)") is fine and never makes a sentence U; keep such names in fixes.
Judge only against the passages, never against your own knowledge."""


class SentenceVerdict(BaseModel):
    n: int = Field(description="the sentence number")
    v: Literal["S", "N", "U"]
    fix: str = Field(default="", description='for U only: the supported rewrite, or ""')


class Verdicts(BaseModel):
    sentences: list[SentenceVerdict]


@dataclass
class ClaimChange:
    sentence: str
    replacement: str  # "" when removed


@dataclass
class ClaimCheck:
    answer: str
    supported: int = 0
    unsupported: int = 0
    changes: list[ClaimChange] = field(default_factory=list)
    declined: bool = False  # replaced by the "couldn't find this" reply
    checked: bool = False  # False when the check couldn't run
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def verified(self) -> bool:
        """Every claim was supported as written (and there was at least one)."""
        return self.checked and not self.declined and self.supported > 0 and self.unsupported == 0


# A sentence ends at . ! or ? (and any closing quote or bracket), then a space
# and a capital letter or an opening quote/bracket, so "1 Ne. 5:13" and
# "600 B.C. in" aren't cut.
_SENTENCE_END = re.compile(r"[.!?][”\"')]*(\s+)(?=[“\"(A-Z])")


def split_sentences(answer: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence; a line (e.g. a list item) is never joined to the next."""
    spans = []
    for line in re.finditer(r"[^\n]+", answer):
        start = line.start()
        for end_match in _SENTENCE_END.finditer(line.group()):
            end = line.start() + end_match.start(1)
            if answer[start:end].strip():
                spans.append((start, end))
            start = line.start() + end_match.end(1)
        if answer[start: line.end()].strip():
            spans.append((start, line.end()))
    return spans


def declined(answer: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


async def check_claims(answer: str, passages: list[str], model, suggestion: str | None = None) -> ClaimCheck:
    """Check the answer's sentences against the passages (see the module docstring).

    model: a chat model (temperature 0). suggestion: a related question for
    the "couldn't find this" reply if the answer has to be replaced.
    """
    result = ClaimCheck(answer=answer)
    spans = split_sentences(answer)
    if not spans or not passages or declined(answer):
        return result
    numbered_passages = "\n\n".join(f"[P{i}]\n{text}" for i, text in enumerate(passages, start=1))
    numbered_sentences = "\n".join(f"{n}. {answer[s:e].strip()}" for n, (s, e) in enumerate(spans, start=1))
    messages = [
        ("system", CHECK_PROMPT),
        ("human", f"Passages:\n{numbered_passages}\n\nAnswer sentences:\n{numbered_sentences}"),
    ]
    started = time.perf_counter()
    try:
        checker = model.with_structured_output(Verdicts, include_raw=True)
        response = await asyncio.wait_for(checker.ainvoke(messages), timeout=CHECK_TIMEOUT_SECONDS)
        verdicts: Verdicts | None = response["parsed"]
        if verdicts is None:
            raise ValueError(f"unparseable claim check: {response.get('parsing_error')}")
    except Exception:  # noqa: BLE001 - never block the answer on the check
        result.seconds = time.perf_counter() - started
        logger.warning("Claim check failed after %.1fs; answer left unverified.", result.seconds, exc_info=True)
        return result
    result.seconds = time.perf_counter() - started
    usage = getattr(response["raw"], "usage_metadata", None) or {}
    result.input_tokens, result.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    result.checked = True

    by_number = {v.n: v for v in verdicts.sentences}
    edits = []
    for n, (start, end) in enumerate(spans, start=1):
        verdict = by_number.get(n)
        if verdict is None or verdict.v == "N":
            continue
        if verdict.v == "S":
            result.supported += 1
            continue
        result.unsupported += 1
        sentence = answer[start:end].strip()
        replacement = verdict.fix.strip()
        edits.append((start, end, replacement))
        result.changes.append(ClaimChange(sentence, replacement))
        logger.info("Unsupported claim %s: %r -> %r", "rewritten" if replacement else "removed", sentence, replacement)

    if result.unsupported > (result.supported + result.unsupported) / 2:
        result.declined = True
        result.answer = not_found_reply(suggestion)
        logger.info("Most claims unsupported (%d of %d): answer replaced by the not-found reply.",
                    result.unsupported, result.supported + result.unsupported)
        return result
    text = answer
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    # Tidy what removals leave behind: doubled spaces, empty lines, a space before punctuation.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    result.answer = text.strip()
    return result


def not_found_reply(suggestion: str | None = None) -> str:
    if suggestion:
        return f"{NOT_FOUND} You could try asking about {suggestion} instead."
    return NOT_FOUND
