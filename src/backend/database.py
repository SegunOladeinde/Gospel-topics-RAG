import os
from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
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
