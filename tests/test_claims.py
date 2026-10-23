"""Tests for src/backend/claims.py and the claim check in finalize_answer().

The checker model is faked: it returns the verdicts each test gives.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.backend.main as backend
from src.backend.claims import NOT_FOUND, Verdicts, check_claims, declined, not_found_reply, split_sentences

PASSAGES = ["Emma Hale Smith married Joseph Smith in 1827. She was the first president of the Relief Society."]


class FakeChecker:
    """A chat model whose structured output is the given verdicts (or raises / hangs)."""

    def __init__(self, verdicts=None, error=None, delay=0.0):
        self.verdicts, self.error, self.delay, self.calls = verdicts, error, delay, []

    def with_structured_output(self, schema, include_raw=False):
        assert schema is Verdicts and include_raw
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        raw = SimpleNamespace(usage_metadata={"input_tokens": 900, "output_tokens": 60})
        return {"parsed": Verdicts(sentences=self.verdicts), "raw": raw, "parsing_error": None}


def verdicts(*items):
    return [{"n": n, "v": v, "fix": fix} for n, v, fix in items]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_sentences_are_split_without_cutting_references_or_lines():
    text = 'Nephi got the plates (1 Ne. 5:10–13). He left around 600 B.C. with his family.\n- A "quoted line." Then more!'
    assert [text[s:e] for s, e in split_sentences(text)] == [
        "Nephi got the plates (1 Ne. 5:10–13).", "He left around 600 B.C. with his family.",
        '- A "quoted line."', "Then more!",
    ]


@pytest.mark.anyio
async def test_supported_answer_is_verified_and_unchanged():
    answer = "Emma married Joseph Smith in 1827. She led the Relief Society. Keep praying about it!"
    checker = FakeChecker(verdicts((1, "S", ""), (2, "S", ""), (3, "N", "")))
    check = await check_claims(answer, PASSAGES, checker)
    assert check.answer == answer and check.verified
    assert (check.supported, check.unsupported, check.input_tokens, check.output_tokens) == (2, 0, 900, 60)
    prompt = checker.calls[0][1][1]
    assert "[P1]\nEmma Hale Smith" in prompt and "3. Keep praying about it!" in prompt


@pytest.mark.anyio
async def test_unsupported_claims_are_rewritten_or_removed_and_logged(caplog):
    answer = "Emma married Joseph in 1827. She was born in 1810. She wrote many hymns. She led the Relief Society."
    checker = FakeChecker(verdicts((1, "S", ""), (2, "U", ""), (3, "U", "She was involved with hymns."),
                                   (4, "S", "")))
    with caplog.at_level(logging.INFO):
        check = await check_claims(answer, PASSAGES, checker)
    assert check.answer == "Emma married Joseph in 1827. She was involved with hymns. She led the Relief Society."
    assert not check.verified and not check.declined
    assert [(c.sentence, c.replacement) for c in check.changes] == [
        ("She was born in 1810.", ""), ("She wrote many hymns.", "She was involved with hymns.")]
    assert "Unsupported claim removed: 'She was born in 1810.'" in caplog.text


@pytest.mark.anyio
async def test_mostly_unsupported_answer_becomes_the_not_found_reply():
    answer = "Lagos has 15 million people. It is in Nigeria. Emma married Joseph."
    check = await check_claims(answer, PASSAGES, FakeChecker(verdicts((1, "U", ""), (2, "U", ""), (3, "S", ""))),
                               suggestion="Church History Topics: Emma Hale Smith")
    assert check.declined and not check.verified
    assert check.answer == f"{NOT_FOUND} You could try asking about Church History Topics: Emma Hale Smith instead."


@pytest.mark.anyio
async def test_a_failed_or_slow_check_leaves_the_answer_unverified(monkeypatch):
    answer = "Emma married Joseph in 1827."
    failed = await check_claims(answer, PASSAGES, FakeChecker(error=RuntimeError("down")))
    assert failed.answer == answer and not failed.checked and not failed.verified
    monkeypatch.setattr("src.backend.claims.CHECK_TIMEOUT_SECONDS", 0.01)
    slow = await check_claims(answer, PASSAGES, FakeChecker(verdicts((1, "S", "")), delay=1))
    assert slow.answer == answer and not slow.checked


@pytest.mark.anyio
async def test_not_found_replies_are_not_checked():
    checker = FakeChecker(verdicts())
    check = await check_claims(not_found_reply("Tithing"), PASSAGES, checker)
    assert declined(check.answer) and not checker.calls and not check.verified


@pytest.mark.anyio
async def test_finalize_shows_no_sources_for_a_not_found_reply_and_checks_the_used_passages():
    docs = [Document(page_content=text, metadata={"source": "church_history_topics", "topic": slug})
            for text, slug in ((PASSAGES[0], "emma-hale-smith"), ("Liberty Jail.", "liberty-jail"))]
    prepared = backend.PreparedAnswer(intent="RAG_NEW_TOPIC", sources=[], documents=docs)

    not_found = await backend.finalize_answer(f"{NOT_FOUND} Try asking about Emma Smith.\nUSED: none",
                                              prepared, FakeChecker())
    assert not_found.sources == [] and not_found.links == [] and not not_found.verified

    checker = FakeChecker(verdicts((1, "S", "")))
    final = await backend.finalize_answer("Emma married Joseph Smith in 1827.\nUSED: [1]", prepared, checker)
    assert final.verified and final.sources == ["church_history_topics/emma-hale-smith"]
    assert "Liberty Jail" not in checker.calls[0][1][1]  # only the used passage is checked against


def test_verified_is_saved_and_returned_with_the_chat(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.backend.database import Base
    from src.backend.history import get_conversation, save_turn

    engine = create_engine(f"sqlite:///{tmp_path / 'chat.db'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        conversation_id = save_turn(db, "u1", None, "Who was Emma Smith?", "Emma married Joseph.", [], verified=True)
        db.commit()
        messages = get_conversation(db, "u1", conversation_id).messages
    assert [(m.role, m.verified) for m in messages] == [("user", False), ("assistant", True)]
