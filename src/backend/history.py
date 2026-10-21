"""Saved chats for signed-in users (chat_conversations + chat_messages).

Every function takes the Clerk user id and only ever touches that user's
conversations, so one user can't read or append to another user's chat by
guessing an id. The user id itself is trusted because it arrives from the
Next.js server alongside the API key (see verify_api_key in main.py).
"""

from collections.abc import Sequence
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.backend.database import ChatConversation, ChatMessage
from src.backend.source_links import SourceLink, source_link

TITLE_MAX_CHARS = 80
LIST_LIMIT = 50


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: datetime


class StoredMessage(BaseModel):
    role: str
    content: str
    sources: list[str]
    source_links: list[SourceLink]


class ConversationDetail(ConversationSummary):
    messages: list[StoredMessage]


def _stored_source(source: str | SourceLink) -> str | dict:
    """What goes in chat_messages.sources: the id, or the full link when known
    (so a reopened chat shows the same titles and URLs as when it was asked)."""
    return source.model_dump() if isinstance(source, SourceLink) else source


def _source_id(entry: str | dict) -> str:
    return entry["id"] if isinstance(entry, dict) else entry


def _link(entry: str | dict) -> SourceLink:
    # Chats saved before links were stored only have ids.
    return SourceLink(**entry) if isinstance(entry, dict) else source_link(entry)


def make_title(question: str) -> str:
    """First line of the first question, collapsed and cut to TITLE_MAX_CHARS."""
    title = " ".join(question.split())
    if len(title) <= TITLE_MAX_CHARS:
        return title
    return title[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def _owned(db: Session, user_id: str, conversation_id: str) -> ChatConversation | None:
    return (
        db.query(ChatConversation)
        .filter(ChatConversation.id == conversation_id, ChatConversation.user_id == user_id)
        .one_or_none()
    )


def save_turn(
    db: Session,
    user_id: str,
    conversation_id: str | None,
    question: str,
    answer: str,
    sources: Sequence[str | SourceLink],
) -> str:
    """Append one question + answer, creating the conversation if needed.

    An id that doesn't exist or belongs to someone else starts a new
    conversation instead of failing the answer the user already paid for.
    Returns the conversation id. Commits; raises SQLAlchemyError on failure.
    """
    now = datetime.now(timezone.utc)
    conversation = _owned(db, user_id, conversation_id) if conversation_id else None
    if conversation is None:
        conversation = ChatConversation(
            user_id=user_id, title=make_title(question), created_at=now, updated_at=now
        )
        db.add(conversation)
        db.flush()  # assigns conversation.id
    conversation.updated_at = now
    # Both share a timestamp; get_conversation() orders "user" before
    # "assistant" on ties.
    db.add_all(
        [
            ChatMessage(conversation_id=conversation.id, role="user", content=question, sources=[], created_at=now),
            ChatMessage(
                conversation_id=conversation.id, role="assistant", content=answer,
                sources=[_stored_source(source) for source in sources], created_at=now,
            ),
        ]
    )
    db.commit()
    return conversation.id


def list_conversations(db: Session, user_id: str) -> list[ConversationSummary]:
    rows = (
        db.query(ChatConversation)
        .filter(ChatConversation.user_id == user_id)
        .order_by(ChatConversation.updated_at.desc())
        .limit(LIST_LIMIT)
        .all()
    )
    return [ConversationSummary(id=row.id, title=row.title, updated_at=row.updated_at) for row in rows]


def get_conversation(db: Session, user_id: str, conversation_id: str) -> ConversationDetail | None:
    conversation = _owned(db, user_id, conversation_id)
    if conversation is None:
        return None
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at, ChatMessage.role.desc())  # "user" before "assistant" on ties
        .all()
    )
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        updated_at=conversation.updated_at,
        messages=[
            StoredMessage(
                role=message.role,
                content=message.content,
                sources=[_source_id(entry) for entry in message.sources or []],
                source_links=[_link(entry) for entry in message.sources or []],
            )
            for message in messages
        ],
    )



def delete_conversation(db: Session, user_id: str, conversation_id: str) -> bool:
    """Delete one of the user's conversations and its messages.

    Returns False if it doesn't exist or belongs to someone else. Commits;
    raises SQLAlchemyError on failure.
    """
    conversation = _owned(db, user_id, conversation_id)
    if conversation is None:
        return False
    # Delete messages explicitly: SQLite (tests) doesn't enforce ON DELETE CASCADE.
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).delete()
    db.delete(conversation)
    db.commit()
    return True
