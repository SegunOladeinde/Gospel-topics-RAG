import logging
import os
from functools import lru_cache

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Index
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

# DATABASE_URL is read lazily in get_session_factory(), after main.py has
# called load_dotenv(), so this module no longer loads .env itself.
logger = logging.getLogger(__name__)

# Supabase's transaction-mode pooler (Supavisor) listens on this port.
SUPABASE_TRANSACTION_POOLER_PORT = 6543


def normalize_database_url(url: str) -> str:
    """Point Postgres URLs at the psycopg 3 driver.

    A bare ``postgresql://`` URL makes SQLAlchemy look for psycopg2, which
    isn't installed, and ``postgres://`` (what Supabase/Heroku hand out) isn't
    a dialect name SQLAlchemy knows at all. Anything else (e.g. sqlite) is
    returned unchanged.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def engine_kwargs(url: str) -> dict:
    """create_engine() options for the given (already normalised) URL."""
    if not url.startswith("postgresql"):
        return {}
    kwargs = {
        # Supabase drops idle connections; test each one before handing it out
        # instead of failing the first query after a quiet period.
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 5,
        "pool_recycle": 1800,
        # Give up on an unreachable database after 5s instead of the OS TCP
        # timeout, so /health and the rate-limit check can't hang on it.
        "connect_args": {"connect_timeout": 5},
    }
    if make_url(url).port == SUPABASE_TRANSACTION_POOLER_PORT:
        # The transaction pooler hands each transaction to whichever server
        # connection is free, so a statement psycopg prepared on one backend
        # doesn't exist on the next and fails with "prepared statement ...
        # does not exist". prepare_threshold=None turns server-side prepared
        # statements off entirely.
        kwargs["connect_args"]["prepare_threshold"] = None
    return kwargs


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker | None:
    """Build the engine + session factory on first call, then reuse it.

    Called from the FastAPI lifespan rather than at import time, so a missing
    or broken DATABASE_URL can't stop the module (and the whole app) from
    loading. Returns None when DATABASE_URL isn't set, which turns
    database-backed rate limiting off. create_engine() doesn't open a
    connection, so this never touches the network.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        logger.warning(
            "DATABASE_URL is not set -- database-backed rate limiting is "
            "DISABLED. Questions will still be answered."
        )
        return None
    url = normalize_database_url(url)
    engine = create_engine(url, **engine_kwargs(url))
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

class UserQueryLog(Base):
    __tablename__ = "user_query_logs"
    # Covers both rate-limit queries in main.py: guests filter on
    # (user_identifier, is_guest), signed-in users add created_at >= cutoff.
    # Keep in sync with sql/001_create_user_query_logs.sql.
    __table_args__ = (
        Index("ix_user_query_logs_rate_limit", "user_identifier", "is_guest", "created_at"),
    )

    id = Column(String, primary_key=True, default=lambda: os.urandom(16).hex())
    user_identifier = Column(String, index=True, nullable=False)
    is_guest = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db() -> sessionmaker | None:
    """Set up the database at startup: build the engine, create missing tables.

    Idempotent -- create_all() only creates tables that don't exist yet.
    Never raises, so a bad DATABASE_URL or an unreachable database can't
    stop the app from booting:
    - invalid URL -> logged, returns None (rate limiting disabled)
    - database unreachable -> logged, the session factory is still returned
      so each request retries and the RATE_LIMIT_FAIL_CLOSED policy applies
    """
    try:
        session_factory = get_session_factory()
    except Exception:  # noqa: BLE001 - e.g. unparseable URL or unknown driver
        logger.exception(
            "Could not create a database engine from DATABASE_URL -- "
            "database-backed rate limiting is DISABLED."
        )
        return None
    if session_factory is None:
        return None
    try:
        Base.metadata.create_all(bind=session_factory.kw["bind"])
        logger.info("Database ready: user_query_logs table exists.")
    except SQLAlchemyError:
        logger.exception(
            "Could not create database tables at startup; rate-limit checks "
            "will retry on each request."
        )
    return session_factory
