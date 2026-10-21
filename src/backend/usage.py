"""How many questions someone has left, and when more become available.

The single source of truth for the question limits: the /api/v1/query limit
check and GET /api/v1/usage both call get_usage(), so the remaining count the
UI shows always matches what the limit check enforces.

- Guests: GUEST_LIMIT questions in total, per IP. It never resets; signing in
  is the way to keep going.
- Signed-in users: USER_LIMIT questions in any rolling USER_WINDOW. A slot
  frees up when the oldest question in the window turns USER_WINDOW old, so
  reset_at is that moment.
- Admins: no limit. Their logged queries (is_admin = true) never count, so
  someone who stops being an admin starts with a full allowance.
"""

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.backend.database import UserQueryLog

GUEST_LIMIT = 1
USER_LIMIT = 7
USER_WINDOW = timedelta(hours=24)


class Usage(BaseModel):
    """Question allowance for one caller. limit/remaining are None for admins."""

    is_guest: bool
    is_admin: bool
    limit: int | None
    used: int
    remaining: int | None
    # When the next question becomes available, if the caller is out of
    # questions and waiting will help. None for guests (no reset), admins,
    # and anyone with questions left.
    reset_at: datetime | None


def _as_utc(moment: datetime) -> datetime:
    # SQLite hands timestamps back without a timezone; Postgres keeps it.
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def get_usage(
    db: Session,
    identifier: str,
    *,
    is_guest: bool,
    is_admin: bool,
    now: datetime | None = None,
) -> Usage:
    """Count the caller's questions. Raises SQLAlchemyError if the DB fails."""
    now = now or datetime.now(timezone.utc)

    if is_admin:
        return Usage(is_guest=False, is_admin=True, limit=None, used=0, remaining=None, reset_at=None)

    if is_guest:
        used = (
            db.query(UserQueryLog)
            .filter(UserQueryLog.user_identifier == identifier, UserQueryLog.is_guest == True)  # noqa: E712
            .count()
        )
        return Usage(
            is_guest=True,
            is_admin=False,
            limit=GUEST_LIMIT,
            used=used,
            remaining=max(GUEST_LIMIT - used, 0),
            reset_at=None,
        )

    window_start = now - USER_WINDOW
    asked_at = [
        row[0]
        for row in db.query(UserQueryLog.created_at)
        .filter(
            UserQueryLog.user_identifier == identifier,
            UserQueryLog.is_guest == False,  # noqa: E712
            UserQueryLog.is_admin == False,  # noqa: E712
            UserQueryLog.created_at >= window_start,
        )
        .order_by(UserQueryLog.created_at)
        .all()
    ]
    used = len(asked_at)
    remaining = max(USER_LIMIT - used, 0)
    # Out of questions: one becomes available once enough of the oldest ones
    # age out to bring the count back under the limit. (used can exceed the
    # limit if two questions raced past the check together.)
    reset_at = (
        _as_utc(asked_at[used - USER_LIMIT]) + USER_WINDOW
        if remaining == 0 and asked_at[used - USER_LIMIT] is not None
        else None
    )
    return Usage(
        is_guest=False,
        is_admin=False,
        limit=USER_LIMIT,
        used=used,
        remaining=remaining,
        reset_at=reset_at,
    )
