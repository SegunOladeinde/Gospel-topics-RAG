import os
from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

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
    }
    if make_url(url).port == SUPABASE_TRANSACTION_POOLER_PORT:
        # The transaction pooler hands each transaction to whichever server
        # connection is free, so a statement psycopg prepared on one backend
        # doesn't exist on the next and fails with "prepared statement ...
        # does not exist". prepare_threshold=None turns server-side prepared
        # statements off entirely.
        kwargs["connect_args"] = {"prepare_threshold": None}
    return kwargs


DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    DATABASE_URL = normalize_database_url(DATABASE_URL)
    engine = create_engine(DATABASE_URL, **engine_kwargs(DATABASE_URL))
else:
    engine = None
if engine:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    SessionLocal = None

Base = declarative_base()

class UserQueryLog(Base):
    __tablename__ = "user_query_logs"

    id = Column(String, primary_key=True, default=lambda: os.urandom(16).hex())
    user_identifier = Column(String, index=True, nullable=False)
    is_guest = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
