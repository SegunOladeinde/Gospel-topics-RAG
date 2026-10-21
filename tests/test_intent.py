"""Tests for the intent router: classify_intent() and is_pure_small_talk().

The chat model is a mock, so these never call OpenAI.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

import src.backend.main as backend
from src.backend.main import Message, classify_intent, is_pure_small_talk


@pytest.fixture
def anyio_backend():
    # anyio's pytest plugin (installed with FastAPI) runs the async tests.
    return "asyncio"


@pytest.fixture(autouse=True)
def empty_intent_cache():
    backend._INTENT_CACHE.clear()
    yield
    backend._INTENT_CACHE.clear()


def _model(label: str) -> MagicMock:
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content=label))
    return model


def _turn(question: str, answer: str) -> list[Message]:
    return [Message(role="user", content=question), Message(role="assistant", content=answer)]


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


@pytest.mark.anyio
async def test_pure_small_talk_skips_the_model():
    model = _model("RAG_NEW_TOPIC")
    assert await classify_intent("Thanks!", [], model) == "CHITCHAT"
    model.ainvoke.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], "RAG_NEW_TOPIC"),
        (_turn("What is faith?", "Faith is..."), "RAG_FOLLOWUP"),
    ],
)
async def test_mixed_message_labelled_chitchat_goes_to_rag(history, expected):
    """Even if the model says CHITCHAT, a question behind a 'thanks' is answered with retrieval."""
    model = _model("CHITCHAT")
    assert await classify_intent("Thanks! And what about tithing?", history, model) == expected


@pytest.mark.anyio
async def test_model_labels_for_real_questions_are_kept():
    assert await classify_intent("What is tithing?", [], _model("RAG_NEW_TOPIC")) == "RAG_NEW_TOPIC"
    history = _turn("What is tithing?", "Tithing is...")
    assert await classify_intent("Why ten percent?", history, _model("RAG_FOLLOWUP")) == "RAG_FOLLOWUP"


@pytest.mark.anyio
async def test_unexpected_label_defaults_to_new_topic():
    assert await classify_intent("What is faith?", [], _model("BANANA")) == "RAG_NEW_TOPIC"


@pytest.mark.anyio
async def test_cache_reuses_label_for_the_same_recent_context():
    history = _turn("What is faith?", "Faith is...")
    model = _model("RAG_FOLLOWUP")
    await classify_intent("Why does it matter?", history, model)
    await classify_intent("Why does it matter?", history, model)
    assert model.ainvoke.call_count == 1


@pytest.mark.anyio
async def test_cache_key_includes_earlier_turns_not_just_the_last_answer():
    """Same question and same last answer, different earlier conversation: classify again."""
    last = _turn("Tell me more", "Here is more detail.")
    about_faith = _turn("What is faith?", "Faith is...") + last
    about_tithing = _turn("What is tithing?", "Tithing is...") + last
    model = _model("RAG_FOLLOWUP")

    await classify_intent("Why does it matter?", about_faith, model)
    await classify_intent("Why does it matter?", about_tithing, model)
    assert model.ainvoke.call_count == 2


@pytest.mark.anyio
async def test_cache_key_uses_only_recent_turns():
    """Turns older than INTENT_CACHE_TURNS don't split the cache."""
    recent = _turn("What is faith?", "Faith is...") + _turn("Tell me more", "More detail.")
    old_a = _turn("Hello", "Hi!") + recent
    old_b = _turn("Good morning", "Morning!") + recent
    assert len(recent) == backend.INTENT_CACHE_TURNS
    model = _model("RAG_FOLLOWUP")

    await classify_intent("Why does it matter?", old_a, model)
    await classify_intent("Why does it matter?", old_b, model)
    assert model.ainvoke.call_count == 1
