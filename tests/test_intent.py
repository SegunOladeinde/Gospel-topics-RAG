"""Tests for the intent router: route_question(), is_pure_small_talk(),
strip_leading_small_talk(), and how prepare_answer() overlaps routing with
retrieval.

The chat model is a mock, so these never call OpenAI.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

import src.backend.main as backend
from src.backend.main import (
    Message,
    RouteDecision,
    is_pure_small_talk,
    prepare_answer,
    route_question,
    strip_leading_small_talk,
)


@pytest.fixture
def anyio_backend():
    # anyio's pytest plugin (installed with FastAPI) runs the async tests.
    return "asyncio"


@pytest.fixture(autouse=True)
def empty_intent_cache():
    backend._INTENT_CACHE.clear()
    yield
    backend._INTENT_CACHE.clear()


def _model(intent: str = "RAG_FOLLOWUP", standalone: str = "What is the standalone question?", fail: bool = False):
    """A chat model whose structured-output router returns the given decision."""
    router = MagicMock()
    if fail:
        router.ainvoke = AsyncMock(side_effect=RuntimeError("OpenAI is down"))
    else:
        router.ainvoke = AsyncMock(
            return_value=backend._RouterOutput(intent=intent, standalone_question=standalone)
        )
    model = MagicMock()
    model.with_structured_output = MagicMock(return_value=router)
    return model, router


def _turn(question: str, answer: str) -> list[Message]:
    return [Message(role="user", content=question), Message(role="assistant", content=answer)]


# ---------------------------------------------------------------------------
# is_pure_small_talk / strip_leading_small_talk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Thanks!",
        "thank you",
        "Hi, thank you so much!",
        "ok thanks",
        "Great answer, thanks!",
        "Hello there",
        "God bless you!",
        "Thanks so much, bye",
        "  THANK YOU!!  ",
    ],
)
def test_pure_small_talk(message):
    assert is_pure_small_talk(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Thanks! And what about tithing?",
        "Thanks and what about tithing",
        "Thanks. Tell me more about the sacrament",
        "Hello, can you explain baptism",
        "thanks for explaining baptism",
        "Great, why is tithing ten percent",
        "Thanks?",
        "What is faith",
        "",
        "and",
        "so so so",
        "\U0001F44D",
    ],
)
def test_not_pure_small_talk(message):
    assert is_pure_small_talk(message) is False


@pytest.mark.parametrize(
    ("question", "searched"),
    [
        ("Thanks! And what about tithing?", "And what about tithing?"),
        ("Hi there. Thank you so much! What is grace?", "What is grace?"),
        ("What is faith?", "What is faith?"),
        ("Thanks for explaining baptism. What about confirmation?", "Thanks for explaining baptism. What about confirmation?"),
        ("Thanks!", "Thanks!"),  # nothing left to search; kept as-is
    ],
)
def test_strip_leading_small_talk(question, searched):
    assert strip_leading_small_talk(question) == searched


# ---------------------------------------------------------------------------
# route_question
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_pure_small_talk_skips_the_model():
    model, router = _model()
    assert await route_question("Thanks!", _turn("What is faith?", "Faith is..."), model) == RouteDecision("CHITCHAT", "Thanks!")
    router.ainvoke.assert_not_called()


@pytest.mark.anyio
async def test_no_history_means_a_new_topic_without_a_model_call():
    """Nothing earlier to follow up on, so the router needn't ask the model."""
    model, router = _model(intent="RAG_FOLLOWUP")
    decision = await route_question("What is tithing?", [], model)
    assert decision == RouteDecision("RAG_NEW_TOPIC", "What is tithing?")
    router.ainvoke.assert_not_called()


@pytest.mark.anyio
async def test_mixed_message_without_history_is_searched_without_the_thanks():
    model, router = _model()
    decision = await route_question("Thanks! And what about tithing?", [], model)
    assert decision == RouteDecision("RAG_NEW_TOPIC", "And what about tithing?")
    router.ainvoke.assert_not_called()


@pytest.mark.anyio
async def test_followup_is_searched_as_the_standalone_question():
    model, router = _model(intent="RAG_FOLLOWUP", standalone="How can I increase my faith in Jesus Christ?")
    decision = await route_question("How can I increase it?", _turn("What is faith?", "Faith is..."), model)
    assert decision == RouteDecision("RAG_FOLLOWUP", "How can I increase my faith in Jesus Christ?")
    router.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_new_topic_with_history_is_searched_as_asked():
    """So the speculative search started in parallel can be reused."""
    model, _ = _model(intent="RAG_NEW_TOPIC", standalone="What is the Word of Wisdom?")
    decision = await route_question("Thanks. What is the Word of Wisdom?", _turn("What is faith?", "Faith is..."), model)
    assert decision == RouteDecision("RAG_NEW_TOPIC", "What is the Word of Wisdom?")


@pytest.mark.anyio
async def test_chitchat_label_for_a_real_question_is_overridden():
    """Even if the model says CHITCHAT, a question behind a 'thanks' gets retrieval."""
    model, _ = _model(intent="CHITCHAT", standalone="What does the Bible Dictionary say about grace?")
    decision = await route_question(
        "Thanks, that helps. What does the Bible Dictionary say about grace?",
        _turn("What is the Atonement?", "The Atonement is..."),
        model,
    )
    assert decision == RouteDecision("RAG_FOLLOWUP", "What does the Bible Dictionary say about grace?")


@pytest.mark.anyio
async def test_model_error_falls_back_to_a_new_topic():
    model, _ = _model(fail=True)
    decision = await route_question("Why ten percent?", _turn("What is tithing?", "Tithing is..."), model)
    assert decision == RouteDecision("RAG_NEW_TOPIC", "Why ten percent?")


@pytest.mark.anyio
async def test_cache_reuses_the_decision_for_the_same_recent_context():
    history = _turn("What is faith?", "Faith is...")
    model, router = _model()
    await route_question("Why does it matter?", history, model)
    await route_question("Why does it matter?", history, model)
    assert router.ainvoke.await_count == 1


@pytest.mark.anyio
async def test_cache_key_includes_earlier_turns_not_just_the_last_answer():
    """Same question and same last answer, different earlier conversation: route again."""
    last = _turn("Tell me more", "Here is more detail.")
    about_faith = _turn("What is faith?", "Faith is...") + last
    about_tithing = _turn("What is tithing?", "Tithing is...") + last
    model, router = _model()

    await route_question("Why does it matter?", about_faith, model)
    await route_question("Why does it matter?", about_tithing, model)
    assert router.ainvoke.await_count == 2


@pytest.mark.anyio
async def test_cache_key_uses_only_recent_turns():
    """Turns older than INTENT_CACHE_TURNS don't split the cache."""
    recent = _turn("What is faith?", "Faith is...") + _turn("Tell me more", "More detail.")
    old_a = _turn("Hello", "Hi!") + recent
    old_b = _turn("Good morning", "Morning!") + recent
    assert len(recent) == backend.INTENT_CACHE_TURNS
    model, router = _model()

    await route_question("Why does it matter?", old_a, model)
    await route_question("Why does it matter?", old_b, model)
    assert router.ainvoke.await_count == 1


# ---------------------------------------------------------------------------
# prepare_answer: routing and retrieval overlap
# ---------------------------------------------------------------------------

def _retriever():
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(
        return_value=[Document(page_content="Tithing is...", metadata={"source": "topical_guide", "topic": "tithing"})]
    )
    return retriever


@pytest.mark.anyio
async def test_new_topic_reuses_the_speculative_search():
    model, _ = _model(intent="RAG_NEW_TOPIC")
    retriever = _retriever()
    prepared = await prepare_answer("What is tithing?", _turn("What is faith?", "Faith is..."), retriever, model)
    assert prepared.intent == "RAG_NEW_TOPIC"
    assert prepared.sources == ["topical_guide/tithing"]
    retriever.ainvoke.assert_awaited_once_with("What is tithing?")
    assert set(prepared.timings) >= {"route_ms", "retrieval_wait_ms", "prepare_ms"}


@pytest.mark.anyio
async def test_followup_searches_the_standalone_question():
    model, _ = _model(intent="RAG_FOLLOWUP", standalone="Why is tithing ten percent?")
    retriever = _retriever()
    prepared = await prepare_answer("Why ten percent?", _turn("What is tithing?", "Tithing is..."), retriever, model)
    assert prepared.intent == "RAG_FOLLOWUP"
    assert retriever.ainvoke.await_args_list[-1].args == ("Why is tithing ten percent?",)


@pytest.mark.anyio
async def test_first_question_makes_no_router_call_and_one_search():
    model, router = _model()
    retriever = _retriever()
    prepared = await prepare_answer("Thanks! And what about tithing?", [], retriever, model)
    assert prepared.intent == "RAG_NEW_TOPIC"
    router.ainvoke.assert_not_called()
    retriever.ainvoke.assert_awaited_once_with("And what about tithing?")


@pytest.mark.anyio
async def test_chitchat_skips_retrieval():
    model, _ = _model()
    retriever = _retriever()
    prepared = await prepare_answer("Thanks so much!", _turn("What is faith?", "Faith is..."), retriever, model)
    assert prepared.intent == "CHITCHAT"
    assert prepared.sources == []
    retriever.ainvoke.assert_not_called()
